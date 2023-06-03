
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


//
// Functions
//

const cost_of = (set) => (w1:float, w2:float, b:float):float => {
  let result = 0;

  for (let i = 0; i < set.length; i++) {
    const x1:float = set[i][0];
    const x2:float = set[i][1];
    const y:float = forward(x1, x2, w1, w2, b);
    const d = y - set[i][2];
    result += d*d;
  }

  return result / set.length;
}

const forward = (x1:float, x2:float, w1:float, w2:float, b:float):float => {
  return sigmoid(x1 * w1 + x2 * w2 + b);
}

const sigmoid = (x:float):float => {
  return 1 / (1 + Math.exp(-x));
}

const confirm = (title, set, w1, w2, b) => {
  const cost = cost_of(set);

  log.ok(`${title}: ${cost(w1, w2, b)}`);
  log.ok(" x1 |  x2 | exp | act");
  log.ok("----+-----+-----+----");

  for (let ix in set) {
    const [ x1, x2, exp ] = set[ix];
    const act = forward(x1, x2, w1, w2, b);
    const print = exp == act.toFixed(0) ? log.ok : log.err;
    print(` ${x1}  |  ${x2}  |  ${exp}  |  ${act.toFixed(0)}  (${act})`);
  }
}

const train = (label:string, set:TrainingSet, eps:number, rate:number, rounds:number) => {
  let w1 = rand();
  let w2 = rand();
  let b  = rand();

  let cost = cost_of(set);

  console.time(`Training ${label}`);

  for (let i = 0; i < rounds; i++) {
    let c = cost(w1, w2, b);
    let dw1 = (cost(w1 + eps, w2, b) - c) / (eps);
    let dw2 = (cost(w1, w2 + eps, b) - c) / (eps);
    let db  = (cost(w1, w2, b + eps) - c) / (eps);
    w1 -= rate * dw1;
    w2 -= rate * dw2;
    b  -= rate * db;
  }
  console.timeEnd(`Training ${label}`);

  log.info(`cost: ${cost(w1, w2, b).toFixed(6)}, params: ${w1.toFixed(6)}, ${w2.toFixed(6)}, bias: ${b.toFixed(6)}`);
  log.info("------------");

  confirm(label, set, w1, w2, b);
}


// Main

export const main = () => {

  const eps  = 1e-2;
  const rate = 1e-2;

  train("OR",   set_or,   eps, rate, 10000);
  train("AND",  set_and,  eps, rate, 10000);
  train("NAND", set_nand, eps, rate, 10000);

}


