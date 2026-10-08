//! Request scheduling for the read-only practice engine. Poker math is supplied
//! by the same pinned engine adapter in production and a gated adapter in tests.

use preflop_solver::blueprint::neural::{PracticePolicyBatchQuery, PracticePolicyQuery};
use preflop_solver::blueprint::Street;
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::sync::{mpsc, Arc, Mutex};

const POSTFLOP_WORKERS: usize = 2;
const QUEUED_POSTFLOP_QUERIES: usize = 8;
type Target = Option<(u64, usize)>;

struct Job {
    query: PracticePolicyQuery,
    target: Target,
}
enum Event {
    Line(std::io::Result<String>),
    InputClosed,
    Resolved(Target, Value),
}
struct Batch {
    request_id: String,
    results: Vec<Value>,
    remaining: usize,
}

fn error(request_id: Value, message: impl Into<String>) -> Value {
    serde_json::json!({"schema": "hu-practice-continual-resolver-error-v1",
                       "requestId": request_id, "error": message.into()})
}

fn guarded_query(
    query: PracticePolicyQuery,
    evaluate: &impl Fn(PracticePolicyQuery) -> Value,
) -> Value {
    let request_id = query.request_id.clone();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| evaluate(query)))
        .unwrap_or_else(|_| error(request_id.into(), "practice resolver batch worker panicked"))
}

fn write_response(output: &mut impl Write, response: &Value) -> std::io::Result<()> {
    serde_json::to_writer(&mut *output, response)?;
    output.write_all(b"\n")?;
    output.flush()
}

struct Dispatcher<'a, W, F> {
    output: &'a mut W,
    evaluate: &'a F,
    jobs: mpsc::SyncSender<Job>,
    pending_jobs: usize,
    next_batch: u64,
    batches: BTreeMap<u64, Batch>,
}

impl<W: Write, F: Fn(PracticePolicyQuery) -> Value + Sync> Dispatcher<'_, W, F> {
    fn deliver(&mut self, target: Target, response: Value) -> std::io::Result<()> {
        if let Some((id, index)) = target {
            let batch = self.batches.get_mut(&id).expect("registered batch");
            batch.results[index] = response;
            batch.remaining -= 1;
            if batch.remaining == 0 {
                let batch = self.batches.remove(&id).expect("completed batch");
                write_response(
                    self.output,
                    &serde_json::json!({
                        "schema": "hu-practice-continual-resolver-batch-result-v1",
                        "requestId": batch.request_id, "results": batch.results,
                    }),
                )?;
            }
            Ok(())
        } else {
            write_response(self.output, &response)
        }
    }

    fn submit(&mut self, query: PracticePolicyQuery, target: Target) -> std::io::Result<()> {
        // Preflop is the frozen lookup/EV-table path, not a postflop solve.
        // It must remain responsive even when both expensive lanes are busy.
        if query.street == Street::Preflop {
            return self.deliver(target, guarded_query(query, self.evaluate));
        }
        let request_id = query.request_id.clone();
        match self.jobs.try_send(Job { query, target }) {
            Ok(()) => {
                self.pending_jobs += 1;
                Ok(())
            }
            Err(mpsc::TrySendError::Full(_)) => self.deliver(
                target,
                error(
                    request_id.into(),
                    "practice resolver is busy; retry this decision",
                ),
            ),
            Err(mpsc::TrySendError::Disconnected(_)) => self.deliver(
                target,
                error(
                    request_id.into(),
                    "practice resolver workers are unavailable",
                ),
            ),
        }
    }

    fn accept(&mut self, line: &str) -> std::io::Result<()> {
        if line.trim().is_empty() {
            return Ok(());
        }
        let value: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(cause) => {
                return self.deliver(
                    None,
                    error(Value::Null, format!("invalid practice query: {cause}")),
                )
            }
        };
        if value.get("schema").and_then(Value::as_str)
            == Some("hu-practice-continual-resolver-batch-query-v1")
        {
            let batch: PracticePolicyBatchQuery = match serde_json::from_value(value) {
                Ok(batch) => batch,
                Err(cause) => {
                    return self.deliver(
                        None,
                        error(
                            Value::Null,
                            format!("invalid practice query batch: {cause}"),
                        ),
                    )
                }
            };
            if batch.request_id.trim().is_empty() || !(1..=2).contains(&batch.queries.len()) {
                return self.deliver(
                    None,
                    error(
                        batch.request_id.into(),
                        "practice resolver batches require one or two queries",
                    ),
                );
            }
            let group = if batch.stream_results {
                None
            } else {
                self.next_batch += 1;
                self.batches.insert(
                    self.next_batch,
                    Batch {
                        request_id: batch.request_id,
                        results: vec![Value::Null; batch.queries.len()],
                        remaining: batch.queries.len(),
                    },
                );
                Some(self.next_batch)
            };
            for (index, query) in batch.queries.into_iter().enumerate() {
                self.submit(query, group.map(|id| (id, index)))?;
            }
            Ok(())
        } else {
            match serde_json::from_value(value) {
                Ok(query) => self.submit(query, None),
                Err(cause) => self.deliver(
                    None,
                    error(Value::Null, format!("invalid practice query: {cause}")),
                ),
            }
        }
    }
}

fn dispatch_events<W: Write, F: Fn(PracticePolicyQuery) -> Value + Sync>(
    events: mpsc::Receiver<Event>,
    mut dispatcher: Dispatcher<'_, W, F>,
) -> std::io::Result<()> {
    let mut input_closed = false;
    while !input_closed || dispatcher.pending_jobs > 0 {
        match events
            .recv()
            .map_err(|_| std::io::Error::other("practice transport stopped"))?
        {
            Event::Line(line) => dispatcher.accept(&line?)?,
            Event::InputClosed => input_closed = true,
            Event::Resolved(target, response) => {
                dispatcher.pending_jobs -= 1;
                dispatcher.deliver(target, response)?;
            }
        }
    }
    Ok(())
}

pub(super) fn serve(
    input: impl BufRead + Send + 'static,
    output: &mut impl Write,
    evaluate: &(impl Fn(PracticePolicyQuery) -> serde_json::Value + Sync),
) -> std::io::Result<()> {
    // The reader never waits for a solve. Bounded input and heavy-work queues
    // prevent replacing head-of-line blocking with unbounded thread/memory use.
    let (events, incoming) = mpsc::sync_channel(32);
    let input_events = events.clone();
    std::thread::spawn(move || {
        for line in input.lines() {
            let failed = line.is_err();
            if input_events.send(Event::Line(line)).is_err() || failed {
                break;
            }
        }
        let _ = input_events.send(Event::InputClosed);
    });
    let (jobs, queued) = mpsc::sync_channel::<Job>(QUEUED_POSTFLOP_QUERIES);
    let queued = Arc::new(Mutex::new(queued));
    std::thread::scope(|scope| {
        for _ in 0..POSTFLOP_WORKERS {
            let queued = queued.clone();
            let events = events.clone();
            scope.spawn(move || loop {
                let job = queued.lock().expect("practice queue poisoned").recv();
                let Ok(job) = job else { break };
                let result = guarded_query(job.query, evaluate);
                if events.send(Event::Resolved(job.target, result)).is_err() {
                    break;
                }
            });
        }
        // Receiver and job sender drop before joining workers, including on
        // output failure, so blocked sends/receives cannot deadlock shutdown.
        dispatch_events(
            incoming,
            Dispatcher {
                output,
                evaluate,
                jobs,
                pending_jobs: 0,
                next_batch: 0,
                batches: BTreeMap::new(),
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Read};
    use std::os::unix::net::UnixStream;
    use std::sync::{mpsc, Mutex};
    use std::time::Duration;

    fn query(id: &str, street: &str) -> serde_json::Value {
        serde_json::json!({
            "requestId": id, "stateHash": "e".repeat(64), "modelVersion": "fixture", "depthBb": 20,
            "privateCards": [7, 34], "board": [], "street": street, "actor": 0,
            "totalPotBb": 1.5, "stacksBb": [19.5, 19], "streetBetsBb": [0.5, 1],
            "totalCommittedBb": [0.5, 1], "lastFullRaiseBb": 1, "raiseReopened": true, "actions": []
        })
    }

    struct FlushGate {
        output: Vec<u8>,
        release: Option<mpsc::Sender<()>>,
    }

    #[test]
    fn legacy_transport_rejects_cash_identity_in_single_and_batch_queries() {
        let mut cash = query("cash", "preflop");
        cash["rulesSha256"] = Value::String("6".repeat(64));
        for request in [cash.clone(), serde_json::json!({
            "schema": "hu-practice-continual-resolver-batch-query-v1", "requestId": "cash-batch", "queries": [cash]
        })] {
            let input = std::io::Cursor::new(format!("{request}\n").into_bytes());
            let mut output = Vec::new();
            serve(input, &mut output, &|_| panic!("cash request must not enter the Home policy")).unwrap();
            let response: Value = serde_json::from_slice(&output).unwrap();
            assert!(response["error"].as_str().unwrap().contains("unknown field"));
        }
    }
    impl Write for FlushGate {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.output.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            if let Some(release) = self.release.take() {
                release.send(()).unwrap();
                release.send(()).unwrap();
            }
            Ok(())
        }
    }

    #[test]
    fn later_preflop_batch_does_not_wait_for_two_active_postflop_queries() {
        let (input, mut producer) = UnixStream::pair().unwrap();
        let (started, starts) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let wait = Mutex::new(wait);
        let producer = std::thread::spawn(move || {
            let slow = serde_json::json!({
                "schema": "hu-practice-continual-resolver-batch-query-v1", "requestId": "slow-batch",
                "streamResults": true, "queries": [query("slow-a", "flop"), query("slow-b", "flop")]
            });
            writeln!(producer, "{slow}").unwrap();
            for _ in 0..2 {
                starts.recv_timeout(Duration::from_secs(2)).unwrap();
            }
            let fast = serde_json::json!({
                "schema": "hu-practice-continual-resolver-batch-query-v1", "requestId": "later-batch",
                "streamResults": true, "queries": [query("fast", "preflop")]
            });
            writeln!(producer, "{fast}").unwrap();
            producer.shutdown(std::net::Shutdown::Write).unwrap();
            // Retain the read half until serve has consumed the input.
            let mut ignored = Vec::new();
            producer.read_to_end(&mut ignored).unwrap();
        });
        let mut writer = FlushGate {
            output: Vec::new(),
            release: Some(release),
        };
        serve(BufReader::new(input), &mut writer, &|query| {
            if query.request_id.starts_with("slow") {
                started.send(()).unwrap();
                wait.lock()
                    .unwrap()
                    .recv_timeout(Duration::from_millis(500))
                    .expect("later ready decision must flush before the active solves finish");
            }
            serde_json::json!({"requestId": query.request_id})
        })
        .unwrap();
        producer.join().unwrap();
        let lines = String::from_utf8(writer.output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["requestId"], "fast");
        assert!(lines.iter().all(|line| line.get("error").is_none()));
    }

    fn batch_input(stream: Option<bool>, queries: Vec<Value>) -> std::io::Cursor<Vec<u8>> {
        let mut value = serde_json::json!({
            "schema": "hu-practice-continual-resolver-batch-query-v1",
            "requestId": "batch", "queries": queries,
        });
        if let Some(stream) = stream {
            value["streamResults"] = stream.into();
        }
        std::io::Cursor::new(format!("{value}\n").into_bytes())
    }

    fn responses(output: Vec<u8>) -> Vec<Value> {
        String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn within_batch_ready_result_flushes_before_slow_sibling() {
        let (release, wait) = mpsc::channel();
        let wait = Mutex::new(wait);
        let mut writer = FlushGate {
            output: Vec::new(),
            release: Some(release),
        };
        serve(
            batch_input(
                Some(true),
                vec![query("slow", "flop"), query("fast", "preflop")],
            ),
            &mut writer,
            &|query| {
                if query.request_id == "slow" {
                    wait.lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(2))
                        .unwrap();
                }
                serde_json::json!({"requestId": query.request_id})
            },
        )
        .unwrap();
        let lines = responses(writer.output);
        assert_eq!(lines[0]["requestId"], "fast");
        assert_eq!(lines[1]["requestId"], "slow");
        assert!(lines.iter().all(|line| line.get("error").is_none()));
    }

    #[test]
    fn default_batch_retains_envelope_and_input_order() {
        let (release, wait) = mpsc::channel();
        let wait = Mutex::new(wait);
        let mut output = Vec::new();
        serve(
            batch_input(None, vec![query("slow", "flop"), query("fast", "flop")]),
            &mut output,
            &|query| {
                if query.request_id == "slow" {
                    wait.lock().unwrap().recv().unwrap();
                } else {
                    release.send(()).unwrap();
                }
                serde_json::json!({"requestId": query.request_id})
            },
        )
        .unwrap();
        let lines = responses(output);
        assert_eq!(lines.len(), 1);
        assert_eq!(
            lines[0]["schema"],
            "hu-practice-continual-resolver-batch-result-v1"
        );
        assert_eq!(lines[0]["requestId"], "batch");
        assert_eq!(lines[0]["results"][0]["requestId"], "slow");
        assert_eq!(lines[0]["results"][1]["requestId"], "fast");
    }

    #[test]
    fn panic_is_identified_and_does_not_poison_a_sibling() {
        let mut output = Vec::new();
        serve(
            batch_input(
                Some(true),
                vec![query("slow", "flop"), query("fast", "preflop")],
            ),
            &mut output,
            &|query| {
                if query.request_id == "slow" {
                    panic!("fixture failure");
                }
                serde_json::json!({"requestId": query.request_id})
            },
        )
        .unwrap();
        let lines = responses(output);
        assert_eq!(lines.len(), 2);
        assert!(lines
            .iter()
            .any(|line| line["requestId"] == "slow" && line["error"].is_string()));
        assert!(lines
            .iter()
            .any(|line| line["requestId"] == "fast" && line.get("error").is_none()));
    }

    #[test]
    fn invalid_input_is_rejected_and_following_query_still_completes() {
        let input = format!("not json\n{{}}\n{{\"schema\":\"hu-practice-continual-resolver-batch-query-v1\",\"requestId\":\"empty\",\"queries\":[]}}\n{}\n", query("valid", "preflop"));
        let mut output = Vec::new();
        serve(
            std::io::Cursor::new(input.into_bytes()),
            &mut output,
            &|query| serde_json::json!({"requestId":query.request_id}),
        )
        .unwrap();
        let lines = responses(output);
        assert_eq!(lines.len(), 4);
        assert!(lines[..3].iter().all(|line| line["error"].is_string()));
        assert_eq!(lines[2]["requestId"], "empty");
        assert_eq!(lines[3]["requestId"], "valid");
    }

    #[test]
    fn output_failure_closes_the_idle_pool_without_deadlock() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "fixture closed",
                ))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let result = serve(
            batch_input(Some(true), vec![query("ready", "preflop")]),
            &mut Broken,
            &|query| serde_json::json!({"requestId":query.request_id}),
        );
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::BrokenPipe);
    }

    #[test]
    fn heavy_concurrency_and_backlog_are_bounded_without_losing_request_ids() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct ReadyGate(FlushGate);
        impl Write for ReadyGate {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.write(bytes)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                let last = self
                    .0
                    .output
                    .split(|byte| *byte == b'\n')
                    .filter(|line| !line.is_empty())
                    .next_back()
                    .unwrap();
                let value: Value = serde_json::from_slice(last).unwrap();
                if value["requestId"] == "fast" {
                    self.0.flush()?;
                }
                Ok(())
            }
        }
        let (input, mut producer) = UnixStream::pair().unwrap();
        let (started, starts) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let wait = Mutex::new(wait);
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let producer = std::thread::spawn(move || {
            let hold = serde_json::json!({"schema":"hu-practice-continual-resolver-batch-query-v1",
                "requestId":"hold", "streamResults":true,
                "queries":[query("hold-a","flop"),query("hold-b","flop")]});
            writeln!(producer, "{hold}").unwrap();
            for _ in 0..2 {
                starts.recv_timeout(Duration::from_secs(2)).unwrap();
            }
            for index in 0..12 {
                let value = serde_json::json!({"schema":"hu-practice-continual-resolver-batch-query-v1",
                    "requestId":format!("batch-{index}"),"streamResults":true,
                    "queries":[query(&format!("queued-{index}"),"flop")]});
                writeln!(producer, "{value}").unwrap();
            }
            writeln!(producer, "{}", query("fast", "preflop")).unwrap();
            producer.shutdown(std::net::Shutdown::Write).unwrap();
            producer.read_to_end(&mut Vec::new()).unwrap();
        });
        let mut writer = ReadyGate(FlushGate {
            output: Vec::new(),
            release: Some(release),
        });
        serve(BufReader::new(input), &mut writer, &|query| {
            if query.street != Street::Preflop {
                let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(count, Ordering::SeqCst);
                if query.request_id.starts_with("hold") {
                    started.send(()).unwrap();
                    wait.lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(2))
                        .unwrap();
                }
                active.fetch_sub(1, Ordering::SeqCst);
            }
            serde_json::json!({"requestId":query.request_id})
        })
        .unwrap();
        producer.join().unwrap();
        let lines = responses(writer.0.output);
        assert_eq!(peak.load(Ordering::SeqCst), POSTFLOP_WORKERS);
        assert_eq!(lines.len(), 15);
        let busy = lines
            .iter()
            .filter(|line| line["error"].is_string())
            .collect::<Vec<_>>();
        assert_eq!(busy.len(), 4);
        assert!(busy
            .iter()
            .all(|line| line["requestId"].as_str().unwrap().starts_with("queued-")));
        assert!(lines
            .iter()
            .any(|line| line["requestId"] == "fast" && line.get("error").is_none()));
    }
}
