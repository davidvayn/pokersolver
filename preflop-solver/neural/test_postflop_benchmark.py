import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from native_value_dataset import board_family
import run_postflop_benchmark as bench


class BenchmarkTests(unittest.TestCase):
    def test_range_renormalization_noise_only_not_public_state_or_support_changes(self):
        a={'board':[2,12,15],'actor':1,'ranges':[[0.,.25,.75],[.5,.5,0.]]}
        b=json.loads(json.dumps(a));b['ranges'][0][1]+=1.3e-16
        self.assertTrue(bench.same_public_state(a,b))
        for changed in ('board','actor','support','weight'):
            c=json.loads(json.dumps(b))
            if changed=='board':c['board']=[3,12,15]
            if changed=='actor':c['actor']=0
            if changed=='support':c['ranges'][0][0]=1e-16
            if changed=='weight':c['ranges'][0][1]+=.000001
            self.assertFalse(bench.same_public_state(a,c))

    def test_prespecified_texture_balance_and_suit_family_exclusion(self):
        first = bench.choose_spots(set())
        excluded = {tuple(s['family']) for s in first}
        second = bench.choose_spots(excluded)
        self.assertEqual(second, bench.choose_spots(excluded))
        self.assertEqual(len(second),12)
        self.assertEqual(len({tuple(s['family']) for s in second}),12)
        for kind,pot in bench.POTS:
            rows=[s for s in second if s['potType']==kind]
            self.assertEqual({s['texture'] for s in rows},set(bench.TEXTURES))
            for row in rows:
                self.assertEqual(row['startingPotBb'],pot)
                self.assertNotIn(board_family(row['board']),excluded)
                self.assertEqual(bench.texture(row['board']),row['texture'])

    def rows(self):
        spots=bench.choose_spots(set())
        rows=[dict(spot=s['id'],seed=seed,halfSummedGainBb=s['startingPotBb']*.006,
                   percentPot=.6,solveSeconds=2) for s in spots for seed in (100101,100102)]
        return spots,rows

    def test_half_sum_normalization_and_seed_cluster_accounting(self):
        spots,rows=self.rows()
        result=bench.summarize(rows,spots)
        self.assertAlmostEqual(result['meanPercentPot'],.6)
        self.assertEqual(result['spotCount'],12)
        self.assertEqual(result['seedCount'],2)
        self.assertEqual(result['fractionSpotsBelow1Percent'],1)
        self.assertIsNone(result['confidenceUpperBound99'])
        self.assertFalse(result['releaseAccepted'])

    def test_missing_duplicate_nonfinite_and_wrong_pot_cannot_score(self):
        spots,rows=self.rows()
        with self.assertRaises(ValueError): bench.summarize(rows[:-1],spots)
        for change in ({'seed':100101},{'percentPot':1.2},{'halfSummedGainBb':float('nan')},
                       {'halfSummedGainBb':-.1}):
            copy=[dict(r) for r in rows];copy[1].update(change)
            with self.assertRaises(ValueError): bench.summarize(copy,spots)

    def test_completed_worker_replays_without_solving_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);stage=root/'stage';out=root/'value.json'
            def worker(*args):
                out.write_text('{"value": 1}')
                return dict(status='complete',workerElapsedSeconds=1)
            with patch.object(bench,'guarded',side_effect=worker) as mocked:
                one=bench.run_job(Path('/binary'),'test',{},stage,[out],threading.Event())
                two=bench.run_job(Path('/binary'),'test',{},stage,[out],threading.Event())
                self.assertEqual(one,two);self.assertEqual(mocked.call_count,1)
                out.write_text('{}')
                with self.assertRaises(ValueError):
                    bench.run_job(Path('/binary'),'test',{},stage,[out],threading.Event())

    def test_partial_outputs_and_receipt_survive_retry_as_unscored_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);stage=root/'stage';stage.mkdir()
            (stage/'attempt-0').mkdir()
            (stage/'completed.json.tmp').write_text('partial')
            out=root/'value.json';out.write_text('partial')
            def worker(*args):
                out.write_text('{"value": 2}')
                return dict(status='complete',workerElapsedSeconds=1)
            with patch.object(bench,'guarded',side_effect=worker):
                bench.run_job(Path('/binary'),'test',{},stage,[out],threading.Event())
            self.assertEqual((stage/'value.json.interrupted-1').read_text(),'partial')
            self.assertEqual((stage/'completed.interrupted-1.tmp').read_text(),'partial')
            self.assertEqual(json.loads(out.read_text()),{'value':2})


if __name__=='__main__': unittest.main()
