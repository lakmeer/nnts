
import { log, assert, floor } from "./utils.ts";
import { AsciiTable3 } from "./_poly-import.ts";


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

// Alloc: New matrix of given size with it's own buffer (optionally pre-fill with random values)
export const alloc = (rows: int, cols: int, rand = false): Matrix => {
  const mem = new Float32Array(rows * cols);
  if (rand) {
    for (let i = 0; i < rows * cols; i++) {
      mem[i] = Math.random() * 2 - 1;
    }
  }
  return { rows, cols, data: mem };
}

// Alloc: New matrix of given size on top of existing buffer (optionally pre-fill with random values)
export const allocIn = (buffer: ArrayBuffer, offset: int, rows: int, cols: int): Matrix => {
  const mem = new Float32Array(buffer, offset, rows * cols);
  return { rows, cols, data: mem };
}

// Create: New matrix of given size, with particular values
export const create = (rows: int, cols: int, values: Array<float>): Matrix => {
  assert(rows * cols === values.length, "create: incompatible dimensions");
  const mem = new Float32Array(values);
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

// Row From: new Matrix of size (1, X) with values from row n of a of [>=n, X]
export const row = (a: Matrix, n: int): Matrix => {
  assert(n < a.rows, "row: row index out of bounds");
  const m = alloc(1, a.cols);
  m.data.set(a.data.subarray(n * a.cols, (n + 1) * a.cols));
  return m;
}

// Submatrix: new Matrix from rectangular selection inside target matrix
export const sub = (m: Matrix, row: int, col: int, rows: int, cols: int): Matrix => {
  const dst = alloc(rows, cols);
  for (let i = 0; i < rows; i++) {
    const rowSel = m.data.subarray((row + i) * m.cols + col, (row + i) * m.cols + col + cols);
    dst.data.set(rowSel, i * cols);
  }
  return dst;
}

// Map-Copy: copy values from one matrix into another, but modify them along the way
export const mapCopy = (dst: Matrix, src: Matrix, fn: (a: float, b: float, ix: int) => float) => {
  assert(dst.rows === src.rows && dst.cols === src.cols, "mapCopy: incompatible dimensions");

  for (let i = 0; i < dst.data.length; i++) {
    dst.data[i] = fn(dst.data[i], src.data[i], i);
  }
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

// Add At
// Add value to specific cell in-place
export const addAt = (m: Matrix, row: int, col: int, val: float) => {
  m.data[row * m.cols + col] += val;
}

// Scale At
// Multiply value in specific cell in-place
export const scaleAt = (m: Matrix, row: int, col: int, val: float) => {
  m.data[row * m.cols + col] *= val;
}

// Apply: maps a function over all entries
export const apply = (m: Matrix, fn: (n: float) => float) => {
  m.data.forEach((v, ix) => m.data[ix] = fn(v));
}

// Equal
export const eq = (a: Matrix, b: Matrix): boolean => {
  print(a, 'a');
  print(b, 'b');
  if (a.rows !== b.rows || a.cols !== b.cols) return false;
  for (let i = 0; i < a.data.length; i++) {
    if (a.data[i] !== b.data[i]) return false;
  }
  return true;
}


//
// Mutations
//

export const splitCols = (m: Matrix, colSpec: int[]): Array<Matrix> => {
  const outputs = [];

  for (let c in colSpec) {
    outputs.push(alloc(m.rows, colSpec[c]));
  }

  for (let r = 0; r < m.rows; r++) {
    let col = 0;
    for (let c in colSpec) {
      let colSize = colSpec[c];
      for (let i = 0; i < colSize; i++) {
        put(outputs[c], r, i, at(m, r, col + i));
      }
      col += colSize;
    }
  }

  return outputs;
}

export const shuffleRows = (m: Matrix) => {

  // More-or-less Fisher-Yates:
  // For index i, pick a random index from the range [i, m.rows]
  // Swap values at i and random index
  // Increment i
  // Repeat until i = m.rows

  const temp = alloc(1, m.cols);

  for (let i = 0; i < m.rows; i++) {
    let j = i + floor(Math.random() * (m.rows - i));

    temp.data.set(m.data.subarray(j * m.cols, (j + 1) * m.cols));
    m.data.set(m.data.subarray(i * m.cols, (i + 1) * m.cols), j * m.cols);
    m.data.set(temp.data, i * m.cols);
  }
}


//
// Utility Functions
//

// Print a matrix
export const print = (m: Matrix, label?:string, sigfig = 4) => {
  const t = new AsciiTable3().setStyle('unicode-single');
  if (label) t.setTitle(label);
  for (let i = 0; i < m.rows; i++) {
    let row = m.data.subarray(i * m.cols, (i + 1) * m.cols);
    t.addRow(...Array.from(row).map(i => i.toFixed(sigfig)));
  }
  console.log(t.toString());
}

// Fill with random values
export const rand = (m: Matrix) => {
  for (let i = 0; i < m.data.length; i++) {
    m.data[i] = Math.random();
  }
}

// Smush into a string
export const smush = (m: Matrix, sigfig = 3): string => {
  return Array.from(m.data).map(v => v.toFixed(sigfig)).join(' ');
}

// Get dimensions
export const dim = (m: Matrix): string => `[${m.rows}×${m.cols}]`;

