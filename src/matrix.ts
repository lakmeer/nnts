
import { log, tableCompact } from "./utils.ts";


//
// Matrix
//
// Cosntructor and methods for dealing with matrices
//

export type Matrix = {
  rows: int;
  cols: int;
  es:   Float32Array;
}


// Alloc: New matrix with it's own memory block

export const alloc = (rows: int, cols: int): Matrix => {
  const mem = new Float32Array(rows * cols);
  return { rows, cols, es: mem };
}


// At: get value at a given row and column

export const at = (m: Matrix, row: int, col: int): float => {
  return m.es[row * m.cols + col];
}


// Set: put a value at a given row and column
// ⚠ MUTABLE ⚠

export const set = (m: Matrix, row: int, col: int, val: float) => {
  m.es[row * m.cols + col] = val;
}


// Fill: fill a matrix with a given value
// ⚠ MUTABLE ⚠

export const fill = (m: Matrix, val: float) => {
  m.es.fill(val);
}


// Dot product

export const dot = (a: Matrix, b: Matrix): Matrix => {
  checkDim(a, b);

  const n = a.cols;
  const dst = alloc(a.rows, b.cols);

  for (let i = 0; i < dst.rows; i++) {
    for (let j = 0; j < dst.cols; j++) {
      set(dst, i, j, 0);

      for (let k = 0; k < n; k++) {
        set(dst, i, j, at(dst, i, j) + at(a, i, k) * at(b, k, j));
      }
    }
  }

  return dst;
}


// Sum
// ⚠ MUTABLE ⚠ - results go in matrix A

export const sum = (a: Matrix, b: Matrix) => {
  checkDim(a, b);
  for (let i = 0; i < a.es.length; i++) {
    a.es[i] += b.es[i];
  }
}


// Print a matrix

export const print = (m: Matrix) => {
  const rows = [];

  for (let i = 0; i < m.rows; i++) {
    const row = [];
    for (let j = 0; j < m.cols; j++) {
      row.push(m.es[i * m.cols + j].toFixed(3));
    }
    rows.push(row);
  }

  console.log( tableCompact(rows) );
}


// Check Dimensions are compatible

export const checkDim = (a: Matrix, b: Matrix) => {
  if (a.cols !== b.rows) {
    throw new Error(`Matrix::checkDim - ${a.cols} !== ${b.rows}`);
  }
}


// Fill with random values

export const rand = (m: Matrix) => {
  for (let i = 0; i < m.es.length; i++) {
    m.es[i] = Math.random();
  }
}

