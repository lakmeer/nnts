

//
// NN.ts
//
// 🔪 Cutting edge machine-learning framework
//

import {
  logHelper, costRank, table, last,
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
    if (SHOW_COST_IN_PROGRESS && i % 2000 == 0) {
      const c = cost(net, net.count, ti, to);
      log.quiet(`${costRank(c)} ${c}`);
    }
  }

  const time = performance.now() - start;
  const c = cost(net, net.count, ti, to);
  const rank = costRank(c);
  log.blue(`${rank} Finished in ${time.toFixed(2)}ms`);
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


//
// NN Class
//

class NeuralNetwork {

  private data:Float32Array;

  static clone (label:string, nn:NeuralNetwork) {
    return new NeuralNetwork(label, nn.arch, false);
  }

  static alloc (arch:Arch):MemLayout {
    const [ inputs, ...layers ] = arch;

    const layout = { size: 0 };

    layout.in = [ 0, inputs ];
    layout.size += layout.in[1];

    layout.ws = [];
    layout.bs = [];
    layout.as = [];

    for (let i = 0; i < layers.length; i++) {
      layout.ws.push([ layout.size, layers[i] * arch[i] ]);
      layout.size += layout.ws[i][1];
    }

    for (let i = 0; i < layers.length; i++) {
      layout.bs.push([ layout.size, layers[i] ]);
      layout.size += layout.bs[i][1];
    }

    for (let i = 0; i < layers.length; i++) {
      layout.as.push([ layout.size, layers[i] ]);
      layout.size += layout.as[i][1];
    }

    layout.out = last(layout.as);

    return layout;
  }


  constructor (label:string, arch:Arch, seed = false) {
    this.label = label;
    this.arch  = arch;
    this.count = arch.length - 1;
    this.activation = sigmoid;

    // Allocate buffer
    this.layout = NeuralNetwork.alloc(arch);
    this.data   = new ArrayBuffer(this.layout.size * 4);

    // Array views
    this.values = new Float32Array(this.data);
    this.params = new Float32Array(this.data, this.layout.ws[0][0] * 4, last(this.layout.bs)[0] + last(this.layout.bs)[1] - 1);

    // Gen layer matrices
    this.layers = { ws: [], bs: [], as: [], in: null, out: null };
    this.layers.in  = Mat.allocIn(this.data, this.layout.in[0] * 4, 1, this.layout.in[1]);
    this.layers.out = Mat.allocIn(this.data, this.layout.out[0] * 4, 1, this.layout.out[1]);
    this.layers.as[0] = this.layers.in;

    for (let i = 0; i < this.count; i++) {
      const [ rows, cols ] = [ this.arch[i], this.arch[i+1] ];
      this.layers.ws[i]   = Mat.allocIn(this.data, this.layout.ws[i][0] * 4, rows, cols);
      this.layers.bs[i]   = Mat.allocIn(this.data, this.layout.bs[i][0] * 4, 1, cols);
      this.layers.as[i+1] = Mat.allocIn(this.data, this.layout.as[i][0] * 4, 1, cols);
    }

    // Initialise memory if required
    if (seed) this.randomize()
  }

  debugData (target:Float32Array = this.params) {
    for (let i = 0; i < target.length - 1; i++) target[i] = i;
  }

  randomize () {
    for (let i = 0; i < this.params.length - 1; i++) this.params[i] = Math.random();
  }

  print (label = this.label) {
    log.info(`${label} (${this.count} layers)`);

    for (let i = 0; i < this.count; i++) {
      Mat.print(this.layers.ws[i], `w${i}`);
      Mat.print(this.layers.bs[i], `b${i}`);
    }
  }

  forward (input:Matrix) {
    const net = this.layers;

    Mat.copy(this.layers.in, input);

    for (let i = 0; i < this.count; i++) {
      Mat.dot(net.as[i+1], net.as[i], net.ws[i]);
      Mat.sum(net.as[i+1], net.bs[i]);
      Mat.apply(net.as[i+1], this.activation);
    }
  }

  cost (ti:Matrix, to:Matrix) {
    const n = ti.rows;
    let c = 0;

    for (let i = 0; i < n; i++) {
      this.forward(Mat.row(ti, i));

      for (let j = 0; j < to.cols; j++) {
        const exp = Mat.at(to, i, j);
        const act = Mat.at(this.layers.out, 0, j);
        c += (exp - act) ** 2;
      }
    }    

    return c / n;
  }

  finite_diff (grad:NeuralNetwork, eps:float, ti:Matrix, to:Matrix) {
    const net = this.layers;

    let saved:float;
    const c = this.cost(ti, to);

    for (let i = 0; i < this.count; i++) {
      let m = net.ws[i];
      let g = grad.layers.ws[i];

      for (let j = 0; j < m.rows; j++) {
        for (let k = 0; k < m.cols; k++) {
          saved = Mat.at(m, j, k);
          Mat.put(m, j, k, saved + eps);
          Mat.put(g, j, k, (this.cost(ti, to) - c) / eps);
          Mat.put(m, j, k, saved);
        }
      }

      m = net.bs[i];
      g = grad.layers.bs[i];

      for (let j = 0; j < m.rows; j++) {
        for (let k = 0; k < m.cols; k++) {
          saved = Mat.at(m, j, k);
          Mat.put(m, j, k, saved + eps);
          Mat.put(g, j, k, (this.cost(ti, to) - c) / eps);
          Mat.put(m, j, k, saved);
        }
      }
    }
  }

  learn (grad:NeuralNetwork, rate:float) {
    const net = this.layers;
    const g   = grad.layers;

    for (let i = 0; i < this.count; i++) {
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

  train (eps:float, rate:float, steps:number, ti:Matrix, to:Matrix) {

    const grad = NeuralNetwork.clone("Gradient", this);

    log.info(`Training for ${steps} steps...`);

    const start = performance.now();

    for (let i = 0; i < steps; i++) {
      this.finite_diff(grad, eps, ti, to);
      this.learn(grad, rate);

      if (SHOW_COST_IN_PROGRESS && i % 2000 == 0) {
        const c = this.cost(ti, to);
        log.quiet(`${costRank(c)} ${c}`);
      }
    }

    const time = performance.now() - start;

    const c = this.cost(ti, to);
    log.blue(`${costRank(c)} Finished in ${time.toFixed(2)}ms`);
  }

  confirm (ti:Matrix, to:Matrix) {
    const net = this.layers;

    const c = this.cost(ti, to);

    log.quiet(`Final Cost:`, c);

    const rows = [];
    let pass = true;

    for (let ix = 0; ix < to.rows; ix++) {
      const exp = Mat.at(to, 0, ix);
      this.forward(Mat.row(ti, ix));
      const act = Mat.at(net.out, 0, 0);
      const ok = exp.toString() == act.toFixed(0);
      pass = pass && ok;
      rows.push([
        ok ? green("OK") : red("XX"),
        Mat.at(net.in, 0, 0),
        Mat.at(net.in, 0, 1),
        exp,
        act.toFixed(0)
      ]);
    }

    if (!pass) log.red("⚠️  Failed to converge");
    console.log(table([ 'OK', 'x1', 'x2', 'exp', 'act' ], rows));
    return pass;
  }

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


  // NN Class

  log.green("NEW METHOD");

  const nn = new NeuralNetwork("Xor", [ 2, 2, 1 ], true);

  nn.train(eps, rate, 50000, ti, to);
  nn.confirm(ti, to);


  // NN Struct

  log.red("OLD METHOD");

  const xor = alloc([ 2, 2, 1 ], true);

  train(xor, eps, rate, 50000, ti, to);
  confirm(xor, ti, to);
}

