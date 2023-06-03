
//
// Modelling logic gates
//

log.quiet("Loaded model: Gates");


//
// Training Data Set
//

const set_or = [
  [ 0, 0, 0 ],
  [ 0, 1, 1 ],
  [ 1, 0, 1 ],
  [ 1, 1, 1 ],
]

const set_and = [
  [ 0, 0, 0 ],
  [ 0, 1, 0 ],
  [ 1, 0, 0 ],
  [ 1, 1, 1 ],
]

const set_nand = [
  [ 0, 0, 1 ],
  [ 0, 1, 1 ],
  [ 1, 0, 1 ],
  [ 1, 1, 0 ],
]

const set_xor = [
  [ 0, 0, 1 ],
  [ 0, 1, 0 ],
  [ 1, 0, 0 ],
  [ 1, 1, 1 ],
]


//
// Model Types
//
// Xor is assuming 3-gate model:
//
//  x1---OR( w1,  w2, b) 
//    \ /               \
//     X                 AND(w1, w2, b) -> y
//    / \               /
//  x2---NAND(w1, w2, b)
//
//
// Each junction is modelled explicitly with it's own params

type Gate = {
  w1:float,
  w2:float,
  b:float,
}

type Xor = {
  or_w1:float,
  or_w2:float,
  or_b:float,

  and_w1:float,
  and_w2:float,
  and_b:float,

  nand_w1:float,
  nand_w2:float,
  nand_b:float,
}

type Model = {
  type: string;
  data: Array<TrainingTuple>;
  params: Gate | Xor,
  forward: (x1:float, x2:float) => float,
  cost: (...args:Array<float>) => float,
}


//
// Functions
//

const cost_of = (set, forward) => (...params):float => {
  let result = 0;

  for (let i = 0; i < set.length; i++) {
    const x1:float = set[i][0];
    const x2:float = set[i][1];
    const y:float = forward(x1, x2, ...params);
    const d = y - set[i][2];
    result += d*d;
  }

  return result / set.length;
}

const forward_gate = (x1:float, x2:float, w1:float, w2:float, b:float):float => {
  return sigmoid(x1 * w1 + x2 * w2 + b);
}

const forward_xor = (x1:float, x2:float, ...params:Array<float>) => {
  const [ or_w1, or_w2, or_b, and_w1, and_w2, and_b, nand_w1, nand_w2, nand_b ] = params;
  const or   = forward_gate(x1, x2, or_w1, or_w2, or_b);
  const nand = forward_gate(x1, x2, nand_w1, nand_w2, nand_b);
  return forward_gate(or, nand, and_w1, and_w2, and_b);
}

const sigmoid = (x:float):float => {
  return 1 / (1 + Math.exp(-x));
}

const confirm = (title, set, forward, params):boolean => {
  const cost = cost_of(set, forward);

  let pass = true;

  log.info(`${title}: ${cost(...params)}`);
  log.info(" x1 |  x2 | exp | act");
  log.info("----+-----+-----+----");

  for (let ix in set) {
    const [ x1, x2, exp ] = set[ix];
    const act = forward(x1, x2, ...params);
    const ok = exp == act.toFixed(0);
    const print = ok ? log.ok : log.err;
    pass = pass && ok;
    print(` ${x1}  |  ${x2}  |  ${exp}  |  ${act.toFixed(0)}  (${act})`);
  }

  return pass;
}

const train = (label:string, set:TrainingSet, forward:Function, eps:number, rate:number, rounds:number) => {
  let w1 = rand();
  let w2 = rand();
  let b  = rand();

  let cost = cost_of(set, forward);

  const start = performance.now();

  for (let i = 0; i < rounds; i++) {
    let c = cost(w1, w2, b);
    let dw1 = (cost(w1 + eps, w2, b) - c) / (eps);
    let dw2 = (cost(w1, w2 + eps, b) - c) / (eps);
    let db  = (cost(w1, w2, b + eps) - c) / (eps);
    w1 -= rate * dw1;
    w2 -= rate * dw2;
    b  -= rate * db;
  }

  const time = performance.now() - start;

  console.log('');

  log.info(`Trained ${label} in ${time.toFixed(2)}ms`);
  log.info(`cost: ${cost(w1, w2, b).toFixed(6)}, params: ${w1.toFixed(6)}, ${w2.toFixed(6)}, bias: ${b.toFixed(6)}`);
  log.info("------------");

  const passed = confirm(label, set, forward, [ w1, w2, b ]);

  if (!passed) {
    log.red("Failed");
  } else {
    log.green("Passed");
  }
}


const train_xor = (label:string, set:TrainingSet, forward:Function, eps:number, rate:number, rounds:number) => {

  let params = [ rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand() ];

  let cost = cost_of(set, forward);

  const start = performance.now();

  for (let i = 0; i < rounds; i++) {
    let c = cost(...params);

    for (let p = 0; p < params.length; p++) {
      let dp = (cost(...params.map((v, ix) => ix == p ? v + eps : v)) - c) / (eps);
      params[p] -= rate * dp;
    }
  }

  const time = performance.now() - start;

  console.log('');

  log.info(`Trained ${label} in ${time.toFixed(2)}ms`);
  log.info(`cost: ${cost(...params).toFixed(6)}, params: ${params.map(v => v.toFixed(6)).join(', ')}`);
  log.info("------------");

  const passed = confirm(label, set, forward, params);

  if (!passed) {
    log.red("Failed");
  } else {
    log.green("Passed");
  }
}


// Main

export const main = () => {

  const eps  = 1e-2;
  const rate = 1e-1;

  train("OR",      set_or,   forward_gate, eps, rate, 5000);
  train("AND",     set_and,  forward_gate, eps, rate, 5000);
  train("NAND",    set_nand, forward_gate, eps, rate, 5000);
  train_xor("XOR", set_xor,  forward_xor,  eps, rate, 20000);

}


