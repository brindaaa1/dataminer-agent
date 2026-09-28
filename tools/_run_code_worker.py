"""run_code 沙箱的子进程入口（固定、可信代码，不插值用户代码，避免转义问题）。
用法：worker.py <输入 parquet> <输出 parquet> <用户代码文件> <id 列名>
用户代码必须定义 def compute(df) -> pandas.Series | pandas.DataFrame。"""
import sys

import pandas as pd


def main(inp, outp, code_path, id_col):
    df = pd.read_parquet(inp)
    ns: dict = {}
    exec(compile(open(code_path).read(), code_path, "exec"), ns)          # noqa: S102 —— 这就是逃生通道本身
    compute = ns.get("compute")
    if compute is None:
        raise SystemExit("代码必须定义 def compute(df)")
    out = compute(df)
    if isinstance(out, pd.Series):
        out = out.to_frame(out.name or "feat0")
    out = out.reset_index(drop=True)
    out.insert(0, id_col, df[id_col].values)
    out.to_parquet(outp)


if __name__ == "__main__":
    main(*sys.argv[1:5])
