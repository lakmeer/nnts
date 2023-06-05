
import { logHelper, exp, assert } from "./utils.ts";

const log = logHelper("NN");
log.quiet("Loaded NN");


//
// NN.ts
//
// 🔪 Cutting edge machine-learning framework
//

import * as Mat from "./matrix.ts";
import type { Matrix } from "./matrix.ts";


//
// Activation Library
//

const sigmoid = (x:float):float => 1 / (1 + exp(-x));


// XOR Model

type Xor = {
  in: Matrix;
  w1: Matrix;
  b1: Matrix;
  a1: Matrix;
  w2: Matrix;
  b2: Matrix;
  a2: Matrix;
}

const initXor = ():Xor => {
  const m:Xor = {} as Xor;
  m.in = Mat.alloc(1, 2); // Inputs
  m.w1 = Mat.alloc(2, 2, true); // Layer 1
  m.b1 = Mat.alloc(1, 2, true);
  m.a1 = Mat.alloc(1, 2);
  m.w2 = Mat.alloc(2, 1, true); // Layer 2
  m.b2 = Mat.alloc(1, 1, true);
  m.a2 = Mat.alloc(1, 1);
  return m;
}

const forwardXor = (m:Xor) => {
  Mat.dot(m.a1, m.in, m.w1);  // Flow inputs to layer 1
  Mat.sum(m.a1, m.b1);        // Apply bias
  Mat.apply(m.a1, sigmoid);   // Apply activation

  Mat.dot(m.a2, m.a1, m.w2);  // Flow first activations to layer 2
  Mat.sum(m.a2, m.b2);        // Apply bias
  Mat.apply(m.a2, sigmoid);   // Apply activation
}

const costXor = (m:Xor, ti:Matrix, to:Matrix) => {

  // ti = Training set inputs
  // to = Training set expected outputs
  assert(ti.rows === to.rows, "costXor: training data pairs must have same number of entries");
  assert(to.cols === m.a2.cols, "costXor: output and last activation must have same column size");

  // Cumulative cost error
  let c = 0;

  // Number of training samples
  const n = ti.rows;

  // For each training sample
  for (let i = 0; i < n; i++) {
    const x:Matrix = Mat.rowFrom(ti, i);
    const y:Matrix = Mat.rowFrom(to, i);

    // Load this sample into the model's input layer
    Mat.copy(m.in, x);

    // Run
    forwardXor(m);

    // Number of output columns
    const q = to.cols;

    // For each output column, add the squared error to the cumulative cost
    for (let j = 0; j < q; j++) {
      let d = Mat.at(m.a2, 0, j) - Mat.at(y, 0, j);
      c += d*d;
    }
  }

  // Return average cost
  return c/n;
}


//
// Main
//

export const main = () => {

  log.quiet("NN.ts - main()");

  const xor = initXor();

  // Set inputs
  Mat.set(xor.in, [ 0, 1 ]);

  // Forward
  forwardXor(xor);

  // Output
  log.info("out", Mat.at(xor.a2, 0, 0));
}

