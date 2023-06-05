
import {
  logHelper,
  costRank,
  table,
  floor,
  log10,
  limit,
  exp, 
  assert,
  red,
  green
} from "./utils.ts";

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



// More abstract NN Models

type Arch = Array<int>;

type NN = {
  count: int; // number of layers
  arch: Arch;
  ws: Matrix[];
  bs: Matrix[];
  as: Matrix[];
}

const nn_alloc = (arch:Arch, seed = false) => {
  const net:NN = {
    count: arch.length - 1,
    arch: arch,
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

const nn_train = (net:NN, eps:float, rate:float, steps: int, ti:Matrix, to:Matrix) => {
  const grad = nn_alloc(net.arch);
  const start = performance.now();

  for (let i = 0; i < 20000; i++) {
    nn_finite_diff(net, grad, eps, ti, to);
    nn_learn(net, grad, rate);
  }

  const time = performance.now() - start;
  const c = nn_cost(net, ti, to);
  const rank = costRank(c);
  log.blue(`${rank} Trained for ${steps} steps in ${time.toFixed(2)}ms`);
}

const nn_confirm = (net:NN, ti: Matrix, to:Matrix):boolean => {

  const input  = net.as[0];
  const output = net.as[net.count];
  const c = nn_cost(net, ti, to);

  log.quiet(`Final Cost:`, c);

  const rows = [];
  let pass = true;

  for (let ix = 0; ix < to.rows; ix++) {
    const exp = Mat.at(to, 0, ix);
    Mat.copy(input, Mat.row(ti, ix));
    nn_forward(net);
    const act = Mat.at(output, 0, 0);
    const ok = exp.toString() == act.toFixed(0);
    pass = pass && ok;
    rows.push([ ok ? green("OK") : red("XX"), Mat.at(input, 0, 0), Mat.at(input, 0, 1), exp, act.toFixed(0) ]);
  }

  if (!pass) log.red("⚠️  Failed to converge");
  console.log(table([ 'OK', 'x1', 'x2', 'exp', 'act' ], rows));
  return pass;

}


//
// Main
//

export const main = () => {

  // Training data
  const set_xor = Mat.create(4, 3, [
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0
  ]);

  const ti = Mat.sub(set_xor, 0, 0, 4, 2);
  const to = Mat.sub(set_xor, 0, 2, 4, 1);

  // Network
  const xor  = nn_alloc([ 2, 2, 1 ], true);

  // Training params
  const eps  = 10e-1;
  const rate = 10e-1;

  // Train
  nn_train(xor, eps, rate, 50000, ti, to);
  nn_confirm(xor, ti, to);

}

