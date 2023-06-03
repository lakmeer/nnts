
import C from "chalk";

export const logHelper = (name: string, color: string) => ({
  ok:    (...args) => console.log(C.green(`[${name}]`), ...args),
  info:  (...args) => console.log(C.blue(`[${name}]`), ...args),
  err:   (...args) => console.log(C.red(`[${name}]`), ...args),
  quiet: (...args) => console.log(C.dim(`[${name}]`), ...args.map(a => C.black(a+""))),
})

