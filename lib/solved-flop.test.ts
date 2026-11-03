import { readFileSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { describe, it, expect, vi } from 'vitest';
import catalog from '@/data/solved-spots/flop-catalog.json';
import { decodeFlopShard, validateFlopManifest, type FlopNode } from '@/lib/solved-flop';
import { beginSpotAction, loadSolvedSpot, prepareSpotSession, sessionNode, spotActions, spotCombo, startSpot, SPOT_SEATS, type SpotCatalogEntry } from '@/lib/solved-spots';
import { applyAction, seededRandom } from '@/lib/practice-engine';
import { evaluate } from '@/lib/evaluator';
import type { HandState } from '@/lib/practice-types';

const entry=catalog[0] as SpotCatalogEntry;
const bytes=readFileSync(path.join(process.cwd(),'public',entry.url));
const manifest=validateFlopManifest(JSON.parse(bytes.toString()));
const directory=path.join(process.cwd(),'public',path.dirname(entry.url));
const nodes=new Map<number,FlopNode>();
for(const shard of manifest.shards){
  const b=readFileSync(path.join(directory,`${shard.sha256}.bin`));
  if(createHash('sha256').update(b).digest('hex')!==shard.sha256) throw Error('corrupt fixture shard');
  for(const [id,node] of decodeFlopShard(b,shard,manifest)) nodes.set(id,node);
}

describe('pinned complete flop library',()=>{
  it('bounds every binary shard and rejects malformed headers, rows and metadata',()=>{
    expect(nodes.size).toBe(manifest.treeNodes);
    expect(manifest.shards.every(s=>s.bytes<=65536)).toBe(true);
    const s=manifest.shards.at(-1)!, b=readFileSync(path.join(directory,`${s.sha256}.bin`));
    const corrupt=Buffer.from(b);corrupt[0]=0;
    expect(()=>decodeFlopShard(corrupt,s,manifest)).toThrow();
    expect(()=>decodeFlopShard(b.subarray(0,b.length-1),s,manifest)).toThrow();
    expect(()=>validateFlopManifest({...manifest,root:manifest.treeNodes})).toThrow();
    expect(()=>validateFlopManifest({...manifest,shards:manifest.shards.slice(1)})).toThrow();
    expect(()=>validateFlopManifest({...manifest,verification:{...manifest.verification,accepted:false}})).toThrow();
  });

  it('loads lazily, preserves a failed pending action, retries without applying it twice, and plays all streets',async()=>{
    const calls:string[]=[];let fail=false;
    const fetcher=vi.spyOn(globalThis,'fetch').mockImplementation(async(url)=>{
      calls.push(String(url));if(fail){fail=false;return new Response('',{status:503});}
      return new Response(readFileSync(path.join(process.cwd(),'public',String(url))));
    });
    try {
      const spot=await loadSolvedSpot(entry), first=startSpot(spot,entry.sha256,'big-blind',seededRandom(17));
      expect(calls.length).toBe(2); // manifest + root shard, not 1,957 shards.
      expect(spotActions(spot,first)).toEqual(spotActions(spot,{...first,state:{...first.state,deck:[...first.state.deck].reverse()}}));
      const root=nodes.get(manifest.root)!;
      const remote=root.edges.find(e=>Math.floor(e.child/128)!==Math.floor(manifest.root/128))!;
      expect(remote).toBeDefined();
      const pending=beginSpotAction(spot,first,remote.action!.id);
      const saved=JSON.stringify(pending.state);fail=true;
      await expect(prepareSpotSession(spot,pending)).rejects.toThrow('retry');
      const resumed=await prepareSpotSession(spot,pending);
      expect(JSON.stringify(resumed.state)).toBe(saved);
      expect(resumed.state.actionHistory.length).toBe(1);
      let current=await prepareSpotSession(spot,startSpot(spot,entry.sha256,'big-blind',seededRandom(22)));
      const streets=new Set<string>();
      while(!current.state.terminal){
        streets.add(current.state.street);const choices=spotActions(spot,current);
        const check=choices.find(a=>a.kind==='check');expect(check).toBeDefined();
        current=await prepareSpotSession(spot,beginSpotAction(spot,current,check!.id));
      }
      expect([...streets]).toEqual(['flop','turn','river']);
      expect(current.nodeId).toBeNull();expect(current.state.result?.reason).toBe('showdown');
      expect(new Set([...current.state.board,...Object.values(current.state.holeCards).flat()]).size).toBe(9);
      expect(current.state.stacksBb['big-blind']+current.state.stacksBb['button-small-blind']).toBe(100);
      expect(calls.length).toBeLessThan(30);
    } finally {fetcher.mockRestore();}
  });

  it('reproduces every available EV and both whole-flop information-set responses in TypeScript',()=>{
    const cards=new Map(manifest.ranges.flat().map(([key])=>[key,spotCombo(key)]));
    const equities=new Map<string,number>();
    function equity(a:number,b:number,board:number[]):number {
      const key=`${board.join(',')}|${a}|${b}`;const cached=equities.get(key);if(cached!==undefined)return cached;
      const known=[...board,...cards.get(a)!,...cards.get(b)!];
      const remaining=Array.from({length:52},(_,c)=>c).filter(c=>!known.includes(c));
      let value=0,count=0;
      const settle=(future:number[])=>{value+=Math.sign(evaluate([...board,...future,...cards.get(a)!])-evaluate([...board,...future,...cards.get(b)!]));count++;};
      if(board.length===5)settle([]);
      else if(board.length===4)for(const c of remaining)settle([c]);
      else for(let i=0;i<remaining.length;i++)for(let j=i+1;j<remaining.length;j++)settle([remaining[i],remaining[j]]);
      if(count!==(board.length===3?990:board.length===4?44:1))throw Error('runout count');
      value/=count;equities.set(key,value);return value;
    }
    let error=0,checked=0,rows=0;
    function walk(id:number,hero:number,combo:number,opponents:Array<[number,number]>,br:boolean):number {
      if(opponents.every(([,w])=>w===0))return 0;
      const node=nodes.get(id)!;
      if(node.tag<2)return opponents.reduce((s,[other,w])=>s+w*(node.tag===0?(node.actor===hero?node.investedBb[1-hero]:-node.investedBb[hero])
        :equity(combo,other,node.board)*(node.investedBb[0]+node.investedBb[1])/2+(node.investedBb[1-hero]-node.investedBb[hero])/2),0);
      if(node.tag===2)return node.edges.reduce((s,e)=>{
        if(cards.get(combo)!.includes(e.card!))return s;
        const masked:Array<[number,number]>=opponents.filter(([key])=>!cards.get(key)!.includes(e.card!)).map(([k,w])=>[k,w/(48-node.board.length)]);
        return s+walk(e.child,hero,combo,masked,br);
      },0);
      const values=node.edges.map((edge,a)=>{
        const masked:Array<[number,number]>=node.actor===hero?opponents:opponents.map(([k,w])=>[k,w===0?0:w*node.rows.find(r=>r.combo===k)!.probabilities[a]]);
        return walk(edge.child,hero,combo,masked,br);
      });
      if(node.actor!==hero)return values.reduce((s,v)=>s+v,0);
      if(br)return Math.max(...values);
      const row=node.rows.find(r=>r.combo===combo)!,mass=opponents.reduce((s,[,w])=>s+w,0);
      if(!row.evBb)throw Error('missing reachable EV');
      rows++;checked+=values.length;values.forEach((v,a)=>error=Math.max(error,Math.abs(v/mass-row.evBb![a])));
      return values.reduce((s,v,a)=>s+v*row.probabilities[a],0);
    }
    const profile=[0,0],best=[0,0];let joint=0;
    for(let p=0;p<2;p++)for(const [combo,w]of manifest.ranges[p]){
      const opponents=manifest.ranges[1-p].filter(([key])=>!cards.get(combo)!.some(c=>cards.get(key)!.includes(c)));
      if(p===0)joint+=w*opponents.reduce((s,[,w])=>s+w,0);
      profile[p]+=w*walk(manifest.root,p,combo,opponents,false);best[p]+=w*walk(manifest.root,p,combo,opponents,true);
    }
    for(let p=0;p<2;p++){
      expect(profile[p]/joint).toBeCloseTo(manifest.verification.profileValueBb[p],8);
      expect(best[p]/joint).toBeCloseTo(manifest.verification.bestResponseValueBb[p],8);
    }
    const available=[...nodes.values()].flatMap(n=>n.rows.filter(r=>r.evBb!==null));
    expect(rows).toBe(available.length);expect(checked).toBe(available.reduce((s,r)=>s+r.evBb!.length,0));
    expect(error).toBeLessThan(.0001);expect((best[0]+best[1]-profile[0]-profile[1])/joint).toBeLessThan(.05);
  },120_000);

  it('checks all public betting branches against the independent shared hand engine',()=>{
    const inputSpot={input:manifest.input,ranges:manifest.ranges,rootNodeId:String(manifest.root)} as Parameters<typeof startSpot>[0];
    const seen=new Set<number>();
    function witness(state:HandState,board:number[]):HandState {
      for(const [a]of manifest.ranges[0])for(const [b]of manifest.ranges[1]){
        const holes=[spotCombo(a),spotCombo(b)],known=[...board,...holes.flat()];
        if(new Set(known).size!==known.length)continue;
        return {...state,board,holeCards:{'button-small-blind':holes[0],'big-blind':holes[1]},deck:Array.from({length:52},(_,c)=>c).filter(c=>!known.includes(c))};
      }
      throw Error('no legal structural witness');
    }
    function walk(id:number,state:HandState){
      if(seen.has(id))throw Error('decision history merged');seen.add(id);
      const node=nodes.get(id)!;
      for(let p=0;p<2;p++)if(Math.abs(state.totalCommittedBb[SPOT_SEATS[p]]-node.investedBb[p])>1e-7)throw Error('commitment mismatch');
      if(node.tag<2){
        if(!state.terminal||state.result?.reason!==(node.tag===0?'fold':'showdown'))throw Error('wrong terminal');
        if(Math.abs(state.stacksBb['big-blind']+state.stacksBb['button-small-blind']-100)>1e-7)throw Error('lost chips');return;
      }
      for(let p=0;p<2;p++)if(Math.abs(state.streetBetsBb[SPOT_SEATS[p]]-node.streetBetsBb[p])>1e-7)throw Error('street accounting');
      if(node.tag===2){
        for(const edge of node.edges){const board=[...node.board,edge.card!];const next=witness(state,board);
          walk(edge.child,{...next,street:board.length===4?'turn':'river',toAct:'big-blind',pendingActors:['big-blind','button-small-blind'],lastFullRaiseBb:1,raiseReopened:true});}
      } else {
        if(state.toAct!==SPOT_SEATS[node.actor]||JSON.stringify(state.board)!==JSON.stringify(node.board))throw Error('wrong public state');
        for(const edge of node.edges){const child=nodes.get(edge.child)!;
          const next=applyAction(state,edge.action!,child.tag===2?{onRoundComplete:s=>s}:{});walk(edge.child,next);}
      }
    }
    walk(manifest.root,startSpot(inputSpot,entry.sha256,'big-blind',seededRandom(7)).state);
    expect(seen.size).toBe(manifest.treeNodes);
  },120_000);
});
