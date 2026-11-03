// Static, content-addressed full-flop bundles. No database or remote solve.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, readdirSync, existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const read=p=>JSON.parse(readFileSync(p,'utf8'));
const sha=b=>createHash('sha256').update(b).digest('hex');
const source=path.join(root,'data/solved-spots');
const registry=read(path.join(source,'flop-library.json'));
if(!Array.isArray(registry)||!registry.length||new Set(registry).size!==registry.length||registry.some(id=>!/^[a-z0-9-]{1,80}$/.test(id))) throw Error('Invalid flop registry');
function run(args){const r=spawnSync(path.join(root,'preflop-solver/target/release/preflop-solver'),args,{encoding:'utf8',timeout:600_000,maxBuffer:2_000_000});
  if(r.error||r.status!==0) throw Error(r.error?.message??r.stderr);return JSON.parse(r.stdout);}
const catalogFile=path.join(source,'flop-catalog.json');
if(process.argv.includes('--verify')) {
  const catalog=read(catalogFile);
  if(!isDeepStrictEqual(catalog.map(e=>e.id),registry)) throw Error('Flop catalog/registry mismatch');
  for(const entry of catalog){
    if(entry.format!=='sharded-flop-v1'||!/^\/models\/solved-spots\/[a-z0-9-]+\/[a-f0-9]{64}\.json$/.test(entry.url)) throw Error('Invalid flop URL');
    const file=path.join(root,'public',entry.url), bytes=readFileSync(file), manifest=JSON.parse(bytes);
    if(bytes.length!==entry.bytes||sha(bytes)!==entry.sha256||entry.sha256!==path.basename(file,'.json')||entry.bytes>2_000_000
      ||manifest.input.id!==entry.id||manifest.input.title!==entry.title||manifest.input.depthBb!==entry.depthBb
      ||!isDeepStrictEqual(manifest.input.board,entry.board)||!isDeepStrictEqual(read(path.join(source,`${entry.id}.json`)),manifest.input)) throw Error('Flop identity mismatch');
    const report=run(['solved-flop-verify','--manifest',file]);
    if(!isDeepStrictEqual(report,manifest.verification)||!isDeepStrictEqual(report,entry.verification)||!report.accepted) throw Error('Flop report mismatch');
    console.log(`${entry.id}: ${manifest.shards.length} immutable shards; ${manifest.decisionNodes} decisions; ${report.totalBestResponseGainBb.toFixed(6)}bb/spot`);
  }
} else {
  const temp=mkdtempSync(path.join(tmpdir(),'poker-flop-export-'));
  try {
    const catalog=[];
    for(const id of registry){
      const directory=path.join(temp,id);
      const result=run(['solved-flop-export','--input',path.join(source,`${id}.json`),'--output-dir',directory,'--max-decision-nodes','500000','--max-seconds','600']);
      if(!/^[a-f0-9]{64}\.json$/.test(result.manifest)) throw Error('Invalid exported manifest name');
      const bytes=readFileSync(path.join(directory,result.manifest)), m=JSON.parse(bytes);
      run(['solved-flop-verify','--manifest',path.join(directory,result.manifest)]);
      const target=path.join(root,'public/models/solved-spots',id);mkdirSync(target,{recursive:true});
      for(const name of readdirSync(directory)){
        if(!/^[a-f0-9]{64}\.(json|bin)$/.test(name)) throw Error('Invalid exported filename');
        const content=readFileSync(path.join(directory,name)), file=path.join(target,name);
        if(existsSync(file)){if(sha(readFileSync(file))!==sha(content)) throw Error('Immutable collision');}
        else writeFileSync(file,content,{flag:'wx'});
      }
      catalog.push({id,title:m.input.title,board:m.input.board,depthBb:m.input.depthBb,format:'sharded-flop-v1',
        url:`/models/solved-spots/${id}/${result.manifest}`,sha256:sha(bytes),bytes:bytes.length,verification:m.verification});
      console.log(`${id}: exported ${m.shards.length} shards, ${m.decisionNodes} decisions`);
    }
    writeFileSync(catalogFile,`${JSON.stringify(catalog,null,2)}\n`);
  } finally {rmSync(temp,{recursive:true});}
}
