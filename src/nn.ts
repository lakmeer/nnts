

//
// NN.ts
//
// 🔪 Cutting edge machine-learning framework
//

import {
  logHelper, costRank, table,
  limit, assert, red, green,
  floor, log10, max, exp, 
} from "./utils.ts";

import * as Mat from "./matrix.ts";
import type { Matrix } from "./matrix.ts";


//
// Setup
//

const log = logHelper("NN");
log.quiet("Loaded NN");

const SHOW_COST_IN_PROGRESS = true;


//
// Activation Library
//

const sigmoid = (x:float):float => 1 / (1 + exp(-x));
const relu    = (x:float):float => max(0, x)
const tanh    = (x:float):float => Math.tanh(x);


//
// Network Structure
//

type Arch = Array<int>;

type NN = {
  count: int; // number of layers
  arch: Arch;
  ws: Matrix[];
  bs: Matrix[];
  as: Matrix[];
}



const alloc = (arch:Arch, seed = false) => {
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

const print = (net:NN, label:string) => {
  log.info(`${label} (${net.count} layers)`);
  for (let i = 0; i < net.count; i++) {
    Mat.print(net.ws[i], `w${i}`);
    Mat.print(net.bs[i], `b${i}`);
  }
}

const forward = (net:NN) => {
  for (let i = 0; i < net.count; i++) {
    Mat.dot(net.as[i+1], net.as[i], net.ws[i]);
    Mat.sum(net.as[i+1], net.bs[i]);
    Mat.apply(net.as[i+1], sigmoid);
  }
}

const cost = (net:NN, ti:Matrix, to:Matrix) => {
  const n = ti.rows;
  let c = 0;

  for (let i = 0; i < n; i++) {

    Mat.copy(net.as[0], Mat.row(ti, i));
    forward(net);

    for (let j = 0; j < to.cols; j++) {
      const exp = Mat.at(to, i, j);
      const act = Mat.at(net.as[net.count], 0, j);
      c += (exp - act) ** 2;
    }
  }    

  return c / n;
}

const finite_diff = (net:NN, grad:NN, eps:float, ti:Matrix, to:Matrix) => {
  let saved:float;
  const c = cost(net, ti, to);

  for (let i = 0; i < net.count; i++) {
    let m = net.ws[i];
    let g = grad.ws[i];

    for (let j = 0; j < m.rows; j++) {
      for (let k = 0; k < m.cols; k++) {
        saved = Mat.at(m, j, k);
        Mat.put(m, j, k, saved + eps);
        forward(net);
        Mat.put(g, j, k, (cost(net, ti, to) - c) / eps);
        Mat.put(m, j, k, saved);
      }
    }

    m = net.bs[i];
    g = grad.bs[i];

    for (let j = 0; j < m.rows; j++) {
      for (let k = 0; k < m.cols; k++) {
        saved = Mat.at(m, j, k);
        Mat.put(m, j, k, saved + eps);
        forward(net);
        Mat.put(g, j, k, (cost(net, ti, to) - c) / eps);
        Mat.put(m, j, k, saved);
      }
    }
  }
}

const learn = (net:NN, g:NN, rate:float) => {
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

const train = (net:NN, eps:float, rate:float, steps: int, ti:Matrix, to:Matrix) => {
  const grad = alloc(net.arch);
  const start = performance.now();

  log.info(`Training for ${steps} steps...`);

  for (let i = 0; i < steps; i++) {
    finite_diff(net, grad, eps, ti, to);
    learn(net, grad, rate);
    if (SHOW_COST_IN_PROGRESS && i % 1000 == 0) {
      const c = cost(net, ti, to);
      log.quiet(`${costRank(c)} ${c}`);
    }
  }

  const time = performance.now() - start;
  const c = cost(net, ti, to);
  const rank = costRank(c);
  log.blue(`${rank} Finished in ${time.toFixed(2)}ms`);
}

const confirm = (net:NN, ti: Matrix, to:Matrix):boolean => {
  const input  = net.as[0];
  const output = net.as[net.count];
  const c = cost(net, ti, to);

  log.quiet(`Final Cost:`, c);

  const rows = [];
  let pass = true;

  for (let ix = 0; ix < to.rows; ix++) {
    const exp = Mat.at(to, 0, ix);
    Mat.copy(input, Mat.row(ti, ix));
    forward(net);
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
  const xor  = alloc([ 2, 2, 1 ], true);

  // Training params
  const eps  = 10e-1;
  const rate = 10e-1;

  // Train
  train(xor, eps, rate, 50000, ti, to);
  confirm(xor, ti, to);

}

