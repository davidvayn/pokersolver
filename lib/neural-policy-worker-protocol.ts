import type { NeuralPolicyResult } from '@/lib/neural-policy';
import type {
  HandState,
  NeuralPolicyRuntime,
  OpponentModelSnapshot,
} from '@/lib/practice-types';

export type NeuralWorkerRequest =
  | {
      id: number;
      type: 'load';
      runtime: NeuralPolicyRuntime;
      modelVersion: string;
      depthBb: number;
    }
  | {
      id: number;
      type: 'infer';
      runtime: NeuralPolicyRuntime;
      modelVersion: string;
      depthBb: number;
      state: HandState;
      profile: OpponentModelSnapshot;
      usage: 'grading' | 'opponent';
    };

export type NeuralWorkerResponse =
  | { id: number; ok: true; result: NeuralPolicyResult | null }
  | { id: number; ok: false; error: string };

export interface NeuralPolicyExecutor {
  load(input: {
    runtime: NeuralPolicyRuntime;
    modelVersion: string;
    depthBb: number;
  }): Promise<void>;
  infer(input: {
    runtime: NeuralPolicyRuntime;
    modelVersion: string;
    depthBb: number;
    state: HandState;
    profile: OpponentModelSnapshot;
    usage: 'grading' | 'opponent';
  }): Promise<NeuralPolicyResult>;
}
