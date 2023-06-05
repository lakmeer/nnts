
import { log, assert, AsciiTable3 } from "./utils.ts";


//
// Matrix
//
// Cosntructor and methods for dealing with matrices
//

export type Matrix = {
  rows: int;
  cols: int;
  data: Float32Array;
}



//
// Construction and Initialization
//

// Alloc: New matrix of given size (optionally pre-fill with random values)
export const alloc = (rows: int, cols: int, rand = false): Matrix => {
  const mem = new Float32Array(rows * cols);
  if (rand) {
    for (let i = 0; i < rows * cols; i++) {
      mem[i] = Math.random();
    }
  }
  return { rows, cols, data: mem };
}


// Fill: fill a matrix with a given value
export const fill = (m: Matrix, val: float) => {
  m.data.fill(val);
}


//
// Access
//

// At: get value at a given row and column (alias 'get')
export const at = (m: Matrix, row: int, col: int): float =>
  m.data[row * m.cols + col];
export const get = at;

// Put: put a value at a given row and column
export const put = (m: Matrix, row: int, col: int, val: float) => 
  m.data[row * m.cols + col] = val;

// Set: put all values at a given row and column
export const set = (m: Matrix, values: Array<float>, start = 0) => {
  m.data.set(values, start);
}

// Copy: copy values from one matrix into another
export const copy = (dst: Matrix, src: Matrix) => {
  assert(dst.rows === src.rows && dst.cols === src.cols, "copy: incompatible dimensions");
  dst.data.set(src.data, 0);
}

// Row From: new Matrix of size (1, a.cols) with values from row n of a
export const rowFrom = (a: Matrix, n: int): Matrix => {
  assert(n < a.rows, "rowFrom: row index out of bounds");
  const m = alloc(1, a.cols);
  m.data.set(a.data.subarray(n * a.cols, (n + 1) * a.cols));
  return m;
}


//
// Binary Operations
//

// Dot product
export const dot = (dst: Matrix, a: Matrix, b: Matrix) => {
  assert(a.cols === b.rows, "dot: incompatible dimensions");
  assert(dst.rows === a.rows && dst.cols === b.cols, "dot: incompatible dest matrix");

  for (let i = 0; i < dst.rows; i++) {
    for (let j = 0; j < dst.cols; j++) {
      put(dst, i, j, 0);
      for (let k = 0; k < a.cols; k++) {
        put(dst, i, j, at(dst, i, j) + at(a, i, k) * at(b, k, j));
      }
    }
  }
}

// Sum
export const sum = (a: Matrix, b: Matrix) => {
  assert(a.rows === b.rows && a.cols === b.cols, "sum: incompatible dimensions");
  for (let i = 0; i < a.data.length; i++) {
    a.data[i] += b.data[i];
  }
}

// Apply: maps a function over all entries
export const apply = (m: Matrix, fn: (n: float) => float) => {
  m.data.forEach((v, ix) => m.data[ix] = fn(v));
}


//
// Utility Functions
//

// Print a matrix
export const print = (m: Matrix, label?:string) => {
  const t = new AsciiTable3().setStyle('none');
  if (label) t.setTitle(label);
  for (let i = 0; i < m.rows; i++) {
    let row = m.data.subarray(i * m.cols, (i + 1) * m.cols);
    t.addRow(row as unknown as Array<float>); 
  }
  console.log(t.toString());
}

// Fill with random values
export const rand = (m: Matrix) => {
  for (let i = 0; i < m.data.length; i++) {
    m.data[i] = Math.random();
  }
}

// Get dimensions
export const dim = (m: Matrix): string => `[${m.rows}×${m.cols}]`;

