

//
// NN.ts
//
// 🔪 Cutting edge machine-learning framework
//

import {
  logHelper, costRank, table, last, pad,
  limit, red, green,
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

type MemLayout = {
  ws: [[ int, int ]];
  bs: [[ int, int ]];
  as: [[ int, int ]];
  in:  [ int, int ];
  out: [ int, int ];
  size: int;
}

type NN = {
  count: int; // number of layers
  arch: Arch;
  ws: Matrix[];
  bs: Matrix[];
  as: Matrix[];
}


//
// NN Functions
//

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

const cost = (net:NN, count:int, ti:Matrix, to:Matrix) => {
  const n = ti.rows;
  let c = 0;

  for (let i = 0; i < n; i++) {
    Mat.copy(net.as[0], Mat.row(ti, i));
    forward(net);

    for (let j = 0; j < to.cols; j++) {
      const exp = Mat.at(to, i, j);
      const act = Mat.at(net.as[count], 0, j);
      c += (exp - act) ** 2;
    }
  }    

  return c / n;
}

const finite_diff = (net:NN, grad:NN, eps:float, ti:Matrix, to:Matrix) => {
  let saved:float;
  const c = cost(net, net.count, ti, to);

  for (let i = 0; i < net.count; i++) {
    let m = net.ws[i];
    let g = grad.ws[i];

    for (let j = 0; j < m.rows; j++) {
      for (let k = 0; k < m.cols; k++) {
        saved = Mat.at(m, j, k);
        Mat.put(m, j, k, saved + eps);
        Mat.put(g, j, k, (cost(net, net.count, ti, to) - c) / eps);
        Mat.put(m, j, k, saved);
      }
    }

    m = net.bs[i];
    g = grad.bs[i];

    for (let j = 0; j < m.rows; j++) {
      for (let k = 0; k < m.cols; k++) {
        saved = Mat.at(m, j, k);
        Mat.put(m, j, k, saved + eps);
        Mat.put(g, j, k, (cost(net, net.count, ti, to) - c) / eps);
        Mat.put(m, j, k, saved);
      }
    }
  }
}

const learn = (net:NN, g:NN, rate:float) => {
  for (let i = 0; i < net.count; i++) {
    for (let j = 0; j < net.ws[i].rows; j++) {
      for (let k = 0; k < net.ws[i].cols; k++) {
        Mat.addAt(net.ws[i], j, k, -rate * Mat.at(g.ws[i], j, k));
      }
    }

    for (let j = 0; j < net.bs[i].rows; j++) {
      for (let k = 0; k < net.bs[i].cols; k++) {
        Mat.addAt(net.bs[i], j, k, -rate * Mat.at(g.bs[i], j, k));
      }
    }
  }
} 

const train = (method: Function, net:NN, eps:float, rate:float, steps: int, ti:Matrix, to:Matrix, peekStride = steps/10) => {
  const grad = alloc(net.arch);
  const start = performance.now();

  log.info(`Training for ${steps} steps...`);

  for (let i = 0; i < steps; i++) {
    method(net, grad, eps, ti, to);
    learn(net, grad, rate);
    if (SHOW_COST_IN_PROGRESS && i % peekStride == 0) {
      const c = cost(net, net.count, ti, to);
      log.quiet(`[${pad(6, '#' + i)}] ${costRank(c)} ${c}`);
    }
  }

  const time = performance.now() - start;
  const c = cost(net, net.count, ti, to);
  const rank = costRank(c);
  log.info(`${rank} Finished in ${time.toFixed(2)}ms`);
}


const confirm = (net:NN, ti: Matrix, to:Matrix):boolean => {
  const input  = net.as[0];
  const output = net.as[net.count];
  const c = cost(net, net.count, ti, to);

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


const backprop = (net:NN, grad:NN, eps:float, ti:Matrix, to:Matrix) => {
  const n = ti.rows;

  // i = current sample
  // l = current layer
  // j = current activation
  // k = previous activation

  zero(grad);

  for (let i = 0; i < n; i++) {
    Mat.copy(net.as[0], Mat.row(ti, i));

    forward(net);

    const expect = Mat.row(to, i);
    const actual = net.as[net.count];

    // Clean gradient activations fro previous training sample
    for (let l = 0; l <= net.count; l++) {
      Mat.fill(grad.as[l], 0);
    }

    // Store output delta in gradient's output activation slot
    for (let j = 0; j < to.cols; j++) {
      Mat.put(grad.as[net.count], 0, j, (Mat.at(expect, 0, j) - Mat.at(actual, 0, j)));
    }

    for (let l = net.count; l > 0; l--) { // Dont run on 0 so we can [l-1] safely
      for (let j = 0; j < net.as[l].cols; j++) {
        let a  = Mat.at(net.as[l], 0, j); // Current activation
        let da = Mat.at(grad.as[l], 0, j); // Partial derivative wrt activation
        Mat.addAt(grad.bs[l-1], 0, j, 2*da*a*(1 - a));

        for (let k = 0; k < net.as[l-1].cols; k++) {
          // j = weight matrix col
          // k = weight matrix row
          let pa = Mat.at(net.as[l-1], 0, k); // Previous activation
          let w  = Mat.at(net.ws[l-1], k, j); // Previous layer weight
          Mat.addAt(grad.ws[l-1], k, j, 2*da*a*(1 - a) * pa);
          Mat.addAt(grad.as[l-1], 0, k, 2*da*a*(1 - a) * w); // Summing all weights to current layer
        }
      }
    }
  }

  // Average the computed gradient
  for (let i = 0; i < grad.count; i++) {
    for  (let j = 0; j < grad.ws[i].rows; j++) {
      for (let k = 0; k < grad.ws[i].cols; k++) {
        Mat.scaleAt(grad.ws[i], j, k, -1/n);
      }
    }

    for  (let j = 0; j < grad.bs[i].rows; j++) {
      for (let k = 0; k < grad.bs[i].cols; k++) {
        Mat.scaleAt(grad.bs[i], j, k, -1/n);
      }
    }
  }
    
  return grad;
}


const zero = (net:NN) => {
  for (let i = 0; i < net.count; i++) {
    Mat.fill(net.ws[i], 0);
    Mat.fill(net.bs[i], 0);
    Mat.fill(net.as[i], 0);
  }
  Mat.fill(net.as[net.count], 0);
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

  const eps  = 10e-2;
  const rate = 10e-2;


  // NN Struct

  const xor  = alloc([ 2, 2, 1 ], true);
  train(finite_diff, xor, 0.01, 0.01, 10000, ti, to);
  confirm(xor, ti, to);

  const xor2  = alloc([ 2, 2, 1 ], true);
  train(backprop, xor2, 0, 1, 10000, ti, to);
  confirm(xor2, ti, to);

}

