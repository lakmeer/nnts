
import { Chalk as C, AsciiTable3 } from "./_poly-import.ts";


// Shim C types

declare global {
  type float = number;
  type int   = number;
}


// Strings, Formatting

type LogFn = (...args: any[]) => void;
type LogHlp = { [key: string]: LogFn }

export const logHelper = (name: string):LogHlp => {
  const label = `[${name}]`;
  return {
    ok:    (...args) => console.log(C.green(label), ...args),
    info:  (...args) => console.log(C.blue(label), ...args),
    err:   (...args) => console.log(C.red(label), ...args),
    quiet: (...args) => console.log(C.dim([label, ...args].join(' '))),
    red:   (...args) => console.log(C.red([label, ...args].join(' '))),
    green: (...args) => console.log(C.green([label, ...args].join(' '))),
    blue:  (...args) => console.log(C.blue([label, ...args].join(' '))),
  }
}

export const log = logHelper("NN");
export const { red, green } = C;
export const pad = (len:number, str:string) => str.padStart(len, ' ');


// Arrays

export const last = <T>(arr:Array<T>):T => arr[arr.length - 1];


// Math

export const { min, max, sin, cos, exp, log10, pow, sqrt, abs, floor, PI } = Math;
export const ln = Math.log; // Already used this name

export const rand = ():float => Math.random();
export const limit = (a:number, b:number, n:number) => min(b, max(a, n));
export const lerp = (a:number, b:number, t:number):number => a + (b - a) * t;
export const sigmoid = (x:float):float => 1 / (1 + exp(-x));
export const unbend = (n) => 1 - (1 - n) * (1 - n);


// Text-mode Tables

export const table = (headers:Array<string>, rows:Array<Array<any>>):string => {
  try {
    const t = new AsciiTable3().setStyle('unicode-round');
    for (let i = 0; i < headers.length; i++) t.setAlignCenter(i + 1); 
    t.setHeading(...headers);
    t.addRowMatrix(rows);
    return t.toString();
  } catch (e) {
    return C.dim("(AsciiTable3 failed)");
  }
}


// Colors

export const ORANGE  = [ 245,  147,  34  ];
export const NEUTRAL = [ 232,  234,  235 ];
export const BLUE    = [ 8,    119,  189 ];
export const DARK    = [ 33,   33,   33  ];

export const colorLerp = (a:number[], b:number[], t:number):string =>
  [ lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t) ];

export const rgb = (c:number[]):string => `rgb(${c.join(',')})`;

export const weightColor = (w:number, z = 1, dark = false) => {
  const s = sigmoid(w*z)*2 - 1;
  const midColor = dark ? DARK : NEUTRAL;
  return w < 0
    ? colorLerp(midColor, ORANGE, -s)
    : colorLerp(midColor, BLUE,    s);
}

export const plainColor = (w:number, z = 1) => {
  const s = sigmoid(w*z)*2 - 1;
  return colorLerp(DARK, NEUTRAL, 1 - (1 - s)*(1 - s));
}


// Misc

export const assert = (cond:boolean, msg:string) => {
  if (!cond) throw new Error(msg);
}

export const costRank = (n:float, symbol = false) => {
  const rank = -floor(log10(n));
  return symbol ? [ "🔴", "🟠", "🟡", "🟢", "🔵", "🟣" ][limit(0, 4, rank - 1)] : rank;
}

