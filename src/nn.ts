
import { logHelper, floor, log10, limit, exp, assert, table, red, green } from "./utils.ts";

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

const initXor = (seed:boolean = false):Xor => {
  const m:Xor = {} as Xor;
  m.in = Mat.alloc(1, 2);       // Inputs
  m.w1 = Mat.alloc(2, 2, seed); // Layer 1
  m.b1 = Mat.alloc(1, 2, seed);
  m.a1 = Mat.alloc(1, 2);
  m.w2 = Mat.alloc(2, 1, seed); // Layer 2
  m.b2 = Mat.alloc(1, 1, seed);
  m.a2 = Mat.alloc(1, 1);       // Outputs
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
  assert(ti.rows === to.rows,   "costXor: training data pairs must have same number of entries");
  assert(to.cols === m.a2.cols, "costXor: training output and last activation must have same width");

  // Cumulative cost error
  let c = 0;

  // For each training sample
  for (let i = 0; i < ti.rows; i++) {

    // Load this sample into the model's input layer
    Mat.copy(m.in, Mat.row(ti, i));

    // Run
    forwardXor(m);

    // For each output column, conpare to exected outputs
    // and add the squared error to the cumulative cost
    const expected:Matrix = Mat.row(to, i);

    for (let j = 0; j < to.cols; j++) {
      let d = Mat.at(m.a2, 0, j) - Mat.at(expected, 0, j);
      c += d*d;
    }
  }

  // Return average cost
  return c/ti.rows;
}

const finiteDiffXor = (m:Xor, g:Xor, eps:float, ti:Matrix, to:Matrix) => {
  let saved:float;
  let c = costXor(m, ti, to);

  for (let i = 0; i < m.w1.rows; i++) {
    for (let j = 0; j < m.w1.cols; j++) {
      saved = Mat.at(m.w1, i, j);
      Mat.put(m.w1, i, j, saved + eps);
      Mat.put(g.w1, i, j, (costXor(m, ti, to) - c) / eps);
      Mat.put(m.w1, i, j, saved);
    }
  }

  for (let i = 0; i < m.b1.rows; i++) {
    for (let j = 0; j < m.b1.cols; j++) {
      saved = Mat.at(m.b1, i, j);
      Mat.put(m.b1, i, j, saved + eps);
      Mat.put(g.b1, i, j, (costXor(m, ti, to) - c) / eps);
      Mat.put(m.b1, i, j, saved);
    }
  }

  for (let i = 0; i < m.w2.rows; i++) {
    for (let j = 0; j < m.w2.cols; j++) {
      saved = Mat.at(m.w2, i, j);
      Mat.put(m.w2, i, j, saved + eps);
      Mat.put(g.w2, i, j, (costXor(m, ti, to) - c) / eps);
      Mat.put(m.w2, i, j, saved);
    }
  }

  for (let i = 0; i < m.b2.rows; i++) {
    for (let j = 0; j < m.b2.cols; j++) {
      saved = Mat.at(m.b2, i, j);
      Mat.put(m.b2, i, j, saved + eps);
      Mat.put(g.b2, i, j, (costXor(m, ti, to) - c) / eps);
      Mat.put(m.b2, i, j, saved);
    }
  }
}

const learnXor = (m:Xor, grad:Xor, rate:float) => {
  const nudge = (v:float, d:float) => v - d * rate;
  Mat.mapCopy(m.w1, grad.w1, nudge);
  Mat.mapCopy(m.b1, grad.b1, nudge);
  Mat.mapCopy(m.w2, grad.w2, nudge);
  Mat.mapCopy(m.b2, grad.b2, nudge);
}

const dumpXor = (m:Xor, label:string = "") => {
  Mat.print(m.w1, label + ':w1');
  Mat.print(m.b1, label + ':b1');
  Mat.print(m.a1, label + ':a1');
  Mat.print(m.w2, label + ':w2');
  Mat.print(m.b2, label + ':b2');
  Mat.print(m.a2, label + ':a2');
}

const confirm = (model:Xor, ti: Matrix, to:Matrix):boolean => {
  let pass = true;
  const rows = [];

  for (let ix = 0; ix < to.rows; ix++) {
    const exp = Mat.at(to, 0, ix);
    Mat.copy(model.in, Mat.row(ti, ix));
    forwardXor(model);
    const act = Mat.at(model.a2, 0, 0);
    const ok = exp.toString() == act.toFixed(0);
    pass = pass && ok;
    rows.push([ ok ? green("OK") : red("XX"), Mat.at(model.in, 0, 0), Mat.at(model.in, 0, 1), exp, act.toFixed(0) ]);
  }

  if (!pass) log.red("⚠️  Failed to converge");
  console.log(table([ 'OK', 'x1', 'x2', 'exp', 'act' ], rows));
  return pass;
}

const train = (model:Xor, ti:Matrix, to:Matrix, eps:number, rate:number, steps:number) => {

  // Training loop
  const start = performance.now();

  const grad = initXor();

  for (let i = 0; i < steps; i++) {
    finiteDiffXor(model, grad, eps, ti, to);
    learnXor(model, grad, rate);
  }

  const time = performance.now() - start;

  const c = costXor(model, ti, to);
  const rank = costRank(c);
  log.blue(`${rank} Trained for ${steps} steps in ${time.toFixed(2)}ms`);
  log.quiet(`Final Cost:`, c);

  const passed = confirm(model, ti, to);
}

const costRank = (n:float) => {
  const rank = -floor(log10(n));
  return [ "🔴", "🟠", "🟡", "🟢", "🔵" ][limit(0, 4, rank - 1)];
}



// More abstract NN Models

type NN = {
  count: int; // number of layers
  ws: Matrix[];
  bs: Matrix[];
  as: Matrix[];
}

type Arch = Array<int>;

const nn_alloc = (arch:Arch, seed = false) => {
  const net:NN = {
    count: arch.length - 1,
    ws: [],
    bs: [],
    as: []
  }

  net.as[0] = Mat.alloc(1, arch[0]);

  for (let i = 1; i <= net.count; i++) {
    net.ws[i-1] = Mat.alloc(net.as[i-1].cols, arch[i], seed);
    net.bs[i-1] = Mat.alloc(1, arch[i], seed);
    net.as[i]   = Mat.alloc(1, arch[i]);
  }

  return net;
}

const nn_print = (net:NN, label:string) => {
  log.info(`${label} (${net.count} layers)`);
  for (let i = 0; i < net.count; i++) {
    Mat.print(net.ws[i], `w${i}`);
    Mat.print(net.bs[i], `b${i}`);
  }
}

const nn_forward = (net:NN) => {
  for (let i = 0; i < net.count; i++) {
    Mat.dot(net.as[i+1], net.as[i], net.ws[i]);
    Mat.sum(net.as[i+1], net.bs[i]);
    Mat.apply(net.as[i+1], sigmoid);
  }
}

const nn_cost = (net:NN, ti:Matrix, to:Matrix) => {
  const n = ti.rows;
  let c = 0;

  for (let i = 0; i < n; i++) {

    Mat.copy(net.as[0], Mat.row(ti, i));
    nn_forward(net);

    for (let j = 0; j < to.cols; j++) {
      const exp = Mat.at(to, i, j);
      const act = Mat.at(net.as[net.count], 0, j);
      c += (exp - act) ** 2;
    }
  }    

  return c / n;
}

const nn_finite_diff = (net:NN, grad:NN, eps:float, ti:Matrix, to:Matrix) => {
  let saved:float;
  const c = nn_cost(net, ti, to);

  for (let i = 0; i < net.count; i++) {
    let m = net.ws[i];
    let g = grad.ws[i];

    for (let j = 0; j < m.rows; j++) {
      for (let k = 0; k < m.cols; k++) {
        saved = Mat.at(m, j, k);
        Mat.put(m, j, k, saved + eps);
        nn_forward(net);
        Mat.put(g, j, k, (nn_cost(net, ti, to) - c) / eps);
        Mat.put(m, j, k, saved);
      }
    }

    m = net.bs[i];
    g = grad.bs[i];

    for (let j = 0; j < m.rows; j++) {
      for (let k = 0; k < m.cols; k++) {
        saved = Mat.at(m, j, k);
        Mat.put(m, j, k, saved + eps);
        nn_forward(net);
        Mat.put(g, j, k, (nn_cost(net, ti, to) - c) / eps);
        Mat.put(m, j, k, saved);
      }
    }
  }
}

const nn_learn = (net:NN, g:NN, rate:float) => {
  for (let i = 0; i < net.count; i++) {
    for (let j = 0; j < net.ws[i].rows; j++) {
      for (let k = 0; k < net.ws[i].cols; k++) {
        Mat.put(net.ws[i], j, k, Mat.at(net.ws[i], j, k) - rate * Mat.at(g.ws[i], j, k));
      }
    }

    for (let j = 0; j < net.bs[i].rows; j++) {
      for (let k = 0; k < net.bs[i].cols; k++) {
        Mat.put(net.bs[i], j, k, Mat.at(net.bs[i], j, k) - rate * Mat.at(g.bs[i], j, k));
      }
    }
  }
} 


//
// Main
//

export const main = () => {

  log.quiet("NN.ts - main()");

  // Training data
  const set_xor = Mat.create(4, 3, [
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0
  ]);

  const ti = Mat.sub(set_xor, 0, 0, 4, 2);
  const to = Mat.sub(set_xor, 0, 2, 4, 1);

  const eps  = 10e-1;
  const rate = 10e-1;

  const arch = [ 2, 2, 1 ];
  const xor  = nn_alloc(arch, true);
  const grad = nn_alloc(arch);

  log.info("Cost before:", nn_cost(xor, ti, to));
  for (let i = 0; i < 1000; i++) {
    nn_finite_diff(xor, grad, eps, ti, to);
    nn_learn(xor, grad, rate);
  }

  log.info("Cost after:", nn_cost(xor, ti, to));

}

