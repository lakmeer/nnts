
//
// Examples
//

export const xor = async () => {

  log.blue("Running XOR Example");


  // XOR Training data

  const ti = Mat.create(4, 2, [ 0, 0,    0, 1,    1, 0,    1, 1   ]);
  const to = Mat.create(4, 1, [       1,       0,       0,      1 ]);


  // Train XOR Network

  const xor = NN.alloc([ 2, 2, 1 ], true);

  await train(xor, ti, to, { 
    maxSteps: 100000,
    maxRank:  4,
    epochSize: 10,
    rate: 40
  });


  // Report

  report(xor, ti, to, [ 'a', 'b' ], (inputs, expect, actual) => {
    return [
      Mat.smush(expect, 0) == Mat.smush(actual, 0),
      Mat.at(inputs, 0, 0).toString(),
      Mat.at(inputs, 0, 1).toString(),
      Mat.smush(expect, 0),
      Mat.smush(actual, 0)
    ]
  });
}


//
// Binary Adder
//

export const adder = async (BITS:number) => {

  // Generate training data

  const n = (1<<BITS);
  const rows = n*n;
  const ti = Mat.alloc(rows, BITS*2);
  const to = Mat.alloc(rows, BITS+1);

  for (let i = 0; i < ti.rows; i++) {
    let x = i/n | 0;
    let y = i%n | 0;
    let z = x + y;

    for (let j = 0; j < BITS; j++) {
      Mat.put(ti, i, j,      (x >> j) & 1);
      Mat.put(ti, i, j+BITS, (y >> j) & 1);
      Mat.put(to, i, j,      (z >> j) & 1);
    }
    Mat.put(to, i, BITS, z >= n ? 1 : 0);
  }


  // Train network


  const net = NN.alloc([ BITS*2, 4*BITS, 3*BITS, BITS+1 ], true);

  await train(net, ti, to, {
    maxSteps:  10000,
    maxRank:   4,
    rate:      6,
    epochSize: 20,
    color: 'limegreen'
  });


  // Report

  report(net, ti, to, [ 'x', 'y' ], (inputs, expect, actual) => {
    const x = Mat.smush(Mat.sub(inputs, 0, 0, BITS, 1), 0);
    const y = Mat.smush(Mat.sub(inputs, 0, BITS, BITS, 1), 0);
    const exp = Mat.smush(expect, 0);
    const act = Mat.smush(actual, 0);
    const ok = exp == act;
    return [ ok, x, y, exp, act ];
  });

  return net;
}


