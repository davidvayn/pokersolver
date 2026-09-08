import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from cloud_blueprint_run import CANONICAL_HAND_CLASSES, combo_weight
from run_exact_preflop_checkdown import merge, read_chunk


class ExactKernelTests(unittest.TestCase):
    def fixture(self):
        # A synthetic all-tie partial game exercises the integer contract;
        # it is explicitly never accepted as a complete Hold'em kernel.
        labels=sorted(CANONICAL_HAND_CLASSES)
        pairs=[1]*(169**2)
        return dict(schema="exact-preflop-checkdown-orbit-chunk-v1",complete=False,
            releaseAccepted=False,classes=labels,classMultiplicities=[combo_weight(c) for c in labels],
            compatiblePairs=pairs,weightedFlopPairs=pairs,weightedWinUnits=[990]*(169**2),
            orbits=[dict(index=0,board=[0,1,2],orbitSize=4,seconds=1.0,integerCountsSha256="a"*64)])

    def test_partial_counts_cannot_become_a_complete_kernel(self):
        chunk=self.fixture()
        with self.assertRaises(ValueError): merge([chunk])
        with self.assertRaises(ValueError): merge([chunk,chunk])
        with self.assertRaises(ValueError): merge([])

    def test_chunk_reader_rejects_count_or_provenance_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"chunk.json.gz"
            chunk=self.fixture()
            for corruption in (None,"units","complete","class","orbit"):
                value=copy.deepcopy(chunk)
                if corruption=="units": value["weightedWinUnits"][0]+=1
                elif corruption=="complete": value["complete"]=True
                elif corruption=="class": value["classes"][0]="XX"
                elif corruption=="orbit": value["orbits"][0]["index"]=1755
                with gzip.open(path,"wt") as stream: json.dump(value,stream)
                if corruption is None: self.assertEqual(read_chunk(path),value)
                else:
                    with self.assertRaises(ValueError): read_chunk(path)


if __name__=="__main__": unittest.main()
