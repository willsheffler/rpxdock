import _pickle, rpxdock as rp
from tabulate import tabulate

def main():
   arg = rp.options.get_cli_args()
   for fn in arg.inputs:
      with open(fn, 'rb') as inp:
         result = _pickle.load(inp)
         df = result.data.to_dataframe()
         # print(result)
         # print(result.to_dataframe().to_fwf())
         content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain")
         print(content)

if __name__ == '__main__':
   main()