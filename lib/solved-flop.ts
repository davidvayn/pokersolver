import { applyAction } from '@/lib/practice-engine';
import { declaredRanges, sessionNode, spotCombo, SPOT_SEATS, type SolvedSpot, type SolvedSpotNode,
  type SpotCatalogEntry, type SpotSession, type SpotVerification } from '@/lib/solved-spots';
import type { LegalAction } from '@/lib/practice-types';

export interface FlopShard { start: number; count: number; sha256: string; bytes: number }
export interface FlopManifest {
  schema: 'solved-flop-sharded-v1'; input: SolvedSpot['input']; inputSha256: string;
  frozenPolicySha256: string; frozenActionValuesSha256: string;
  root: number; treeNodes: number; decisionNodes: number; shardSize: number;
  ranges: SolvedSpot['ranges']; shards: FlopShard[]; verification: SpotVerification;
}
export interface FlopNode {
  id: number; tag: number; actor: number; board: number[]; investedBb: [number,number]; streetBetsBb: [number,number];
  edges: Array<{ child: number; card?: number; action?: LegalAction }>;
  rows: SolvedSpotNode['rows'];
}
const hashPattern = /^[a-f0-9]{64}$/;
const int = (n: number, max: number) => Number.isSafeInteger(n) && n >= 0 && n <= max;
const sum = (values: number[]) => values.reduce((a,b) => a+b,0);
const equal = (a: unknown,b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const fail = () => new Error('Invalid saved flop data. No fallback strategy will be used.');

export function validateFlopManifest(value: unknown): FlopManifest {
  try {
    const m=value as FlopManifest, p=m.input, v=m.verification;
    if (m.schema!=='solved-flop-sharded-v1' || ![m.inputSha256,m.frozenPolicySha256,m.frozenActionValuesSha256].every(h=>hashPattern.test(h))
      || !/^[a-z0-9-]{1,80}$/.test(p.id) || typeof p.title!=='string' || typeof p.description!=='string' || typeof p.rangeAssumptions!=='string'
      || p.board.length!==3 || new Set(p.board).size!==3 || p.board.some(c=>!int(c,51))
      || !Number.isFinite(p.depthBb) || p.depthBb<2 || p.depthBb>1000 || !Number.isFinite(p.potBb) || p.potBb<2 || p.potBb>=p.depthBb*2
      || !int(p.raiseCap,1) || !int(p.iterations,1024) || p.iterations<20
      || [p.betPotFractions,p.raisePotFractions].some(s=>!s.length || s.length>2 || s.some((n,i)=>!Number.isFinite(n)||n<=0||n>4||(i>0&&n<=s[i-1])))
      || !int(m.treeNodes,2_000_000) || m.treeNodes<1 || m.root!==m.treeNodes-1
      || !int(m.decisionNodes,500_000) || m.decisionNodes<1 || m.decisionNodes>m.treeNodes || m.shardSize!==128
      || m.shards.length!==Math.ceil(m.treeNodes/128)) throw fail();
    m.shards.forEach((s,i)=>{if(s.start!==i*128 || s.count!==Math.min(128,m.treeNodes-i*128) || !hashPattern.test(s.sha256)
      || !int(s.bytes,65536) || s.bytes<10) throw fail();});
    const ranges=declaredRanges(p);
    if(m.ranges.length!==2) throw fail();
    m.ranges.forEach((r,i)=>{if(!r.length || r.length>32 || r.length!==ranges[i].size || new Set(r.map(([k])=>k)).size!==r.length
      || r.some(([k,w])=>!ranges[i].has(k)||!Number.isFinite(w)||w<=0||Math.abs(w-ranges[i].get(k)!)>1e-12)) throw fail();});
    const deals=m.ranges[0].reduce((n,[a])=>n+m.ranges[1].filter(([b])=>!spotCombo(a).some(c=>spotCombo(b).includes(c))).length,0);
    if(!v.accepted || v.method!=='direct-compatible-pair-and-runout-enumeration-of-exported-f32-policy-v1'
      || v.compatibleDeals!==deals || !deals || v.riverOutcomes!==deals*45*44
      || [v.totalBestResponseGainBb,v.maximumActionEvErrorBb,v.maximumProbabilitySumError,v.zeroSumResidualBb].some(n=>!Number.isFinite(n)||n<0)
      || v.totalBestResponseGainBb>.05 || v.maximumActionEvErrorBb>.0001 || v.maximumProbabilitySumError>1e-6 || v.zeroSumResidualBb>1e-8
      || v.profileValueBb.length!==2 || v.bestResponseValueBb.length!==2 || v.unilateralGainBb.length!==2
      || [...v.profileValueBb,...v.bestResponseValueBb,...v.unilateralGainBb].some(n=>!Number.isFinite(n))
      || Math.abs(sum(v.profileValueBb))>1e-8 || Math.abs(sum(v.unilateralGainBb)-v.totalBestResponseGainBb)>1e-8
      || v.unilateralGainBb.some((n,i)=>n< -1e-8 || Math.abs(n-v.bestResponseValueBb[i]+v.profileValueBb[i])>1e-8)) throw fail();
    return m;
  } catch { throw fail(); }
}

class Reader {
  at=0;
  view: DataView;
  constructor(readonly bytes: Uint8Array) {this.view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);}
  take(n:number) {if(!int(n,this.bytes.length)||this.at+n>this.bytes.length) throw fail();const result=this.bytes.subarray(this.at,this.at+n);this.at+=n;return result;}
  u8() {return this.take(1)[0];}
  u16() {const at=this.at;this.take(2);return this.view.getUint16(at,true);}
  u32() {const at=this.at;this.take(4);return this.view.getUint32(at,true);}
  f32() {const at=this.at;this.take(4);return this.view.getFloat32(at,true);}
  f64() {const at=this.at;this.take(8);return this.view.getFloat64(at,true);}
}
export function decodeFlopShard(bytes:Uint8Array, shard:FlopShard, m:FlopManifest):Map<number,FlopNode> {
  const r=new Reader(bytes), result=new Map<number,FlopNode>();
  if(bytes.length!==shard.bytes || bytes.length>65536 || new TextDecoder().decode(r.take(4))!=='PSF1' || r.u32()!==shard.start || r.u16()!==shard.count) throw fail();
  for(let id=shard.start;id<shard.start+shard.count;id++) {
    const row=new Reader(r.take(r.u32())), tag=row.u8(), actor=row.u8(), length=row.u8(), padded=Array.from(row.take(5)), board=padded.slice(0,length);
    const investedBb:[number,number]=[row.f64(),row.f64()], streetBetsBb:[number,number]=[row.f64(),row.f64()], edges=row.u16(), rows=row.u16();
    if(!int(tag,3) || ![3,4,5].includes(length) || new Set(board).size!==length || board.some((c,i)=>!int(c,51)||(i<3&&c!==m.input.board[i]))
      || padded.slice(length).some(c=>c!==255) || ((tag===0||tag===3)? ![0,1].includes(actor):actor!==255)
      || investedBb.some(n=>!Number.isFinite(n)||n<0||n>m.input.depthBb) || streetBetsBb.some((n,i)=>!Number.isFinite(n)||n<0||n>investedBb[i])
      || ((tag<2)&&(edges!==0||rows!==0)) || (tag===2&&(length===5||edges!==52-length||rows!==0)) || (tag===3&&(edges<1||edges>8||rows>32))) throw fail();
    const node:FlopNode={id,tag,actor,board,investedBb,streetBetsBb,edges:[],rows:[]};
    if(tag===2) for(let i=0;i<edges;i++) node.edges.push({card:row.u8(),child:row.u32()});
    if(tag===3) for(let i=0;i<edges;i++) {
      const n=row.u8();if(n<1||n>80) throw fail();const actionId=new TextDecoder('utf-8',{fatal:true}).decode(row.take(n));
      const kind=row.u8(), target=row.f64(), child=row.u32();
      if(kind>5 || !Number.isFinite(target) || (kind<3 ? target!==0 : target<=streetBetsBb[actor] || target>streetBetsBb[actor]+m.input.depthBb-investedBb[actor]+1e-8)) throw fail();
      const kinds=['fold','check','call','bet','raise','all-in'] as const;
      const label=kind===0?'Fold':kind===1?'Check':kind===2?`Call ${(Math.max(...streetBetsBb)-streetBetsBb[actor]).toFixed(1)}bb`:`${kind===5?'All-in':kind===4?'Raise to':'Bet'} ${target.toFixed(1)}bb`;
      node.edges.push({child,action:{id:actionId,kind:kinds[kind],label,...(kind>=3?{amountToBb:target}:{})}});
    }
    if(node.edges.some(e=>!int(e.child,id-1)) || (tag===2&&(new Set(node.edges.map(e=>e.card)).size!==edges||node.edges.some(e=>!int(e.card!,51)||board.includes(e.card!))))
      || (tag===3&&new Set(node.edges.map(e=>e.action!.id)).size!==edges)) throw fail();
    if(tag===3) {
      const expected=m.ranges[actor].map(([k])=>k).filter(k=>!spotCombo(k).some(c=>board.includes(c))).sort((a,b)=>a-b);
      if(rows!==expected.length) throw fail();
      for(let i=0;i<rows;i++) {
        const combo=row.u16(), available=row.u8(), probabilities=Array.from({length:edges},()=>row.f32());
        const evBb=available===1?Array.from({length:edges},()=>row.f32()):null;
        const total=sum(probabilities);
        if(combo!==expected[i] || available>1 || probabilities.some(n=>!Number.isFinite(n)||n<0||n>1) || Math.abs(total-1)>1e-6
          || evBb?.some(n=>!Number.isFinite(n)||Math.abs(n)>m.input.depthBb+1e-6)) throw fail();
        node.rows.push({combo,probabilities:probabilities.map(n=>n/total),evBb});
      }
    }
    if(row.at!==row.bytes.length) throw fail();result.set(id,node);
  }
  if(r.at!==bytes.length) throw fail();return result;
}

async function readImmutable(url:string, expected:{bytes:number;sha256:string}, signal?:AbortSignal,reload=false):Promise<Uint8Array> {
  const response=await fetch(url,{signal,cache:reload?'reload':'force-cache'});
  if(!response.ok) throw new Error('The saved strategy shard could not be loaded. Please retry.');
  const reader=response.body?.getReader();if(!reader) throw fail();const chunks:Uint8Array[]=[];let size=0;
  try {while(true){const {value,done}=await reader.read();if(done) break;size+=value.length;if(size>expected.bytes) throw fail();chunks.push(value);}}
  catch(error){await reader.cancel().catch(()=>{});throw error;}
  const bytes=new Uint8Array(size);let at=0;for(const chunk of chunks){bytes.set(chunk,at);at+=chunk.length;}
  const sha=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
  if(size!==expected.bytes||sha!==expected.sha256) throw new Error('Saved strategy checksum mismatch. No decision was graded.');return bytes;
}

export async function loadFlopSpot(entry:SpotCatalogEntry, signal?:AbortSignal):Promise<SolvedSpot> {
  if(!/^\/models\/solved-spots\/[a-z0-9-]+\/[a-f0-9]{64}\.json$/.test(entry.url)||!hashPattern.test(entry.sha256)||!int(entry.bytes,2_000_000)||entry.bytes<1) throw fail();
  const m=validateFlopManifest(JSON.parse(new TextDecoder().decode(await readImmutable(entry.url,entry,signal))));
  if(m.input.id!==entry.id||m.input.title!==entry.title||m.input.depthBb!==entry.depthBb||!equal(m.input.board,entry.board)||!equal(m.verification,entry.verification)) throw fail();
  const cached=new Map<number,Map<number,FlopNode>>(), pending=new Map<number,Promise<void>>(), failed=new Set<number>();
  const base=entry.url.slice(0,entry.url.lastIndexOf('/')+1);
  async function ensure(id:number) {
    if(!int(id,m.treeNodes-1)) throw fail();const index=Math.floor(id/128);
    if(cached.has(index)){const v=cached.get(index)!;cached.delete(index);cached.set(index,v);return;}
    if(pending.has(index)) return pending.get(index);
    const shard=m.shards[index];
    const request=readImmutable(`${base}${shard.sha256}.bin`,shard,signal,failed.has(index)).then(bytes=>{
      cached.set(index,decodeFlopShard(bytes,shard,m));failed.delete(index);
      while(cached.size>16) cached.delete(cached.keys().next().value!);
    }).catch(error=>{failed.add(index);throw error;}).finally(()=>pending.delete(index));pending.set(index,request);return request;
  }
  function raw(id:string) {if(!/^(0|[1-9][0-9]*)$/.test(id)) return;return cached.get(Math.floor(Number(id)/128))?.get(Number(id));}
  const spot:SolvedSpot={schema:m.schema,input:m.input,inputSha256:m.inputSha256,rootNodeId:String(m.root),ranges:m.ranges,nodes:[],nodeCount:m.decisionNodes,verification:m.verification};
  spot.runtime={
    node(id) {const n=raw(id);if(!n||n.tag!==3) return;return {id,actor:n.actor as 0|1,board:n.board,investedBb:n.investedBb,streetBetsBb:n.streetBetsBb,rows:n.rows,
      actions:n.edges.map(e=>({...e.action!,nextNodeId:String(e.child)}))};},
    step(session,actionId) {const n=sessionNode(spot,session);const action=n.actions.find(a=>a.id===actionId);if(!action) throw fail();return {state:applyAction(session.state,action),nodeId:action.nextNodeId};},
    async prepare(session) {
      let id=Number(session.nodeId);
      for(let hop=0;hop<3;hop++) {
        if(session.nodeId===null) throw fail();await ensure(id);const n=raw(String(id));if(!n) throw fail();
        if(n.tag===3){const next={state:session.state,nodeId:String(id)};sessionNode(spot,next);return next;}
        if(n.tag<2){if(!session.state.terminal||session.state.result?.reason!==(n.tag===0?'fold':'showdown')
          || !equal(session.state.board.slice(0,n.board.length),n.board)
          || SPOT_SEATS.some((seat,p)=>Math.abs(session.state.totalCommittedBb[seat]-n.investedBb[p])>1e-6)
          || (n.tag===0&&session.state.result.winner!==SPOT_SEATS[n.actor])) throw fail();return {state:session.state,nodeId:null};}
        if(session.state.terminal || session.state.board.length!==n.board.length+1 || !equal(session.state.board.slice(0,n.board.length),n.board)) throw fail();
        const edge=n.edges.find(e=>e.card===session.state.board[n.board.length]);if(!edge) throw fail();id=edge.child;
      }
      throw fail();
    },
  };
  await ensure(m.root);return spot;
}
