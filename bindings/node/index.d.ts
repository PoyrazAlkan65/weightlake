export type DType =
  | 'F64' | 'F32' | 'F16' | 'BF16'
  | 'I64' | 'I32' | 'I16' | 'I8'
  | 'U64' | 'U32' | 'U16' | 'U8'
  | 'BOOL' | 'F8_E4M3' | 'F8_E5M2';

export interface TensorStats {
  mean: number; std: number; l2: number; abs_mean: number;
  min: number; max: number; sparsity: number; nan: number; inf: number;
}

export interface TensorInfo {
  name: string; dtype: DType; shape: number[]; role: string;
  numel: number; nbytes: number; n_chunks: number; stats?: TensorStats;
}

export interface Commit {
  id: string; parents: string[]; message: string; timestamp: number;
  manifest_id: string; n_tensors: number; logical_bytes: number;
}

export interface ImportOptions {
  branch?: string; message?: string; base?: string | null;
  delta?: boolean; compress?: boolean; chunkMiB?: number; includeAux?: boolean;
}

export interface Tensor {
  name: string; dtype: DType; shape: number[];
  data: Float64Array | Float32Array | BigInt64Array | BigUint64Array | Int32Array | Uint32Array
      | Int16Array | Uint16Array | Int8Array | Uint8Array;
}

export class Database {
  readonly path: string;
  importModel(path: string, opts?: ImportOptions): { commit: Commit; stats: Record<string, number>; files: number };
  export(rev: string, outDir: string): string[];
  resolve(rev: string): string;
  log(rev?: string, limit?: number): Commit[];
  branches(): Record<string, string>;
  createBranch(name: string, rev?: string): string;
  deleteBranch(name: string): void;
  tensors(rev?: string): TensorInfo[];
  tensorNames(rev?: string): string[];
  tensorInfo(name: string, rev?: string): TensorInfo;
  readTensor(name: string, rev?: string, opts?: { raw?: boolean }): Tensor;
  readTensorBuffer(name: string, rev?: string): Buffer;
  weight(name: string, index: number, rev?: string): number;
  diff(a: string, b: string, opts?: { numeric?: boolean }): { changes: any[]; unchanged: number };
  sql(query: string): Record<string, unknown>[];
  sqlPretty(query: string): string;
  stats(): Record<string, number>;
  gc(opts?: { dryRun?: boolean }): Record<string, unknown>;
}

export function open(path: string, opts?: { create?: boolean }): Database;
export function init(path: string): Database;
export const version: string;
