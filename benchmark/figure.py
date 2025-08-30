import os
import numpy as np
import csv
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
import argparse
import time
import re
import math

class subPlotdata:
  def __init__(self):
    self.x = list()
    self.y = list()
    self.label = list()

def runTest(FLAG, MAKE_Flag):
  os.popen("rm -rf *.mlir")
  cmd_str = os.popen("make " + FLAG + MAKE_Flag + " 2>&1").read()
  print("result:")
  print(cmd_str)
  if FLAG == "":
      FLAG="default"
  else:
      FLAG = FLAG.replace(" ","_")
  cmd_str = cmd_str.replace(" GFLOPS","")
  cmd_str = cmd_str.replace("\n\n","ENTER")
  cmd_str = cmd_str.replace("\n",", ")
  cmd_str = cmd_str.replace("ENTER","\n")
  with open('data_'+FLAG+'.csv','w') as f:
      f.write(cmd_str)
  CSV_File = 'data_'+FLAG+'.csv'
  return CSV_File

def draw(CSV_File):
  with open(CSV_File, "r") as csvfile: 
    csvReader = csv.reader(csvfile)  # 读取csv文件
    Data = list(csvReader)  # csv数据转换为列表
  hashmap = dict()
  row_len = len(Data)  # 得到数据行数
  for i in range(0, row_len):
    hashmap.setdefault(re.findall("conv-(\d+)-",Data[i][0])[0],[]).append(Data[i])

  plotList = list()
  for i in hashmap:
    sub = subPlotdata()
    for data in hashmap[i]:
        sub.label.append(data[0].replace("conv-",""))
        sub.x.append(float(data[1]))  # 将第一列数据从第二行读取到最后一行赋给列表x
        sub.y.append(float(data[2]))  # 将第二列数据从第二行读取到最后一行赋给列表y
    plotList.append(sub)
  for i in range(1,len(plotList)+1):
    pylab.rcParams.update({'figure.figsize': '26, 24'})
    plt.subplot(math.ceil(len(plotList)/2.0), 2, i)
    idx = np.arange(len(plotList[i-1].x)) *2
    total_width, n = 0.8, 2
    width = total_width / n
    idx = idx - (total_width - width) / 2

    plt.barh(idx, width=plotList[i-1].x,  height=width, label='buddy')
    plt.barh(idx + width, width=plotList[i-1].y, height=width, label='default')
    plt.yticks(idx + width/2,plotList[i-1].label)
  CSV_File = CSV_File.replace("data","")
  CSV_File = CSV_File.replace(".csv","")
  plt.savefig("figure"+CSV_File+".png")

if __name__ == '__main__':
  time_start=time.time()
  parser = argparse.ArgumentParser()
  parser.add_argument("-STRIP", type=int, help="Set the size of each vector slice, default=32")
  parser.add_argument("-FILTER_min", type=int, help="Set the MINIMUM size of the convolution kernel, default=3")
  parser.add_argument("-FILTER_max", type=int, help="Set the MAXIMUM size of the convolution kernel, default=11")
  parser.add_argument("-FILTER_step", type=int, help="Set the step size of the convolution kernel size change, default=2")
  parser.add_argument("-OUTPUT_min", type=int, help="Set the MINIMUM size of the output matrix, default=32")
  parser.add_argument("-OUTPUT_max", type=int, help="Set the MAXIMUM size of the output matrix, default=1024")
  parser.add_argument("-OUTPUT_step", type=int, help="Set the step size of the output matrix size change(OUTPUT+=step), default is OUTPUT*=2")
  parser.add_argument("-CSV", help="Generate figure directly using the given csv file")
  parser.add_argument("-CONV_OPT", help="the path of the conv-opt (MLIR convolution optimizer with CB-SM approach)")
  parser.add_argument("-MLIR_OPT", help="the path of the mlir-opt (MLIR modular optimizer driver).")
  parser.add_argument("-PASS", type=int, help="the number of test passes used to average the results.")
  args = parser.parse_args()

  Flag = ""
  MAKE_Flag = ""
  if (args.STRIP):
    Flag += "STRIP=" + str(args.STRIP) + " "
  if (args.FILTER_min):
    Flag += "FILTER_min=" + str(args.FILTER_min) + " "
  if (args.FILTER_max):
    Flag += "FILTER_max=" + str(args.FILTER_max) + " "
  if (args.FILTER_step):
    Flag += "FILTER_step=" + str(args.FILTER_step) + " "
  if (args.OUTPUT_min):
    Flag += "OUTPUT_min=" + str(args.OUTPUT_min) + " "
  if (args.OUTPUT_max):
    Flag += "OUTPUT_max=" + str(args.OUTPUT_max) + " "
  if (args.OUTPUT_step):
    Flag += "OUTPUT_step=" + str(args.OUTPUT_step) + " "
  if (args.CONV_OPT):
    MAKE_Flag += "CONV_OPT=\"" + str(args.CONV_OPT) + "\" "
  if (args.MLIR_OPT):
    MAKE_Flag += "MLIR_OPT=\"" + str(args.MLIR_OPT) + "\" "

  CSV_File=""
  print("running...")
  if args.CSV == None:
    CSV_File = runTest(Flag, MAKE_Flag)
    if args.PASS and args.PASS > 1:
      #处理第一次结果
      with open(CSV_File, "r") as csvfile: 
        csvReader = csv.reader(csvfile)  
        Data = list(csvReader) 
      hashmap = dict()
      row_len = len(Data) 
      for i in range(0, row_len):
        hashmap.setdefault(Data[i][0],[]).append(float(Data[i][1]))
        hashmap[Data[i][0]].append(float(Data[i][2]))
      # 之后每次的结果累加
      for i in range(1,args.PASS):
        runTest(Flag, MAKE_Flag)
        with open(CSV_File, "r") as csvfile: 
          csvReader = csv.reader(csvfile)  # 读取csv文件
          Data = list(csvReader)  # csv数据转换为列表
        row_len = len(Data)  # 得到数据行数
        for i in range(0, row_len):
          hashmap[Data[i][0]][0] += float(Data[i][1])
          hashmap[Data[i][0]][1] += float(Data[i][2])
      #加和除n取算数平均，写入csv
      with open(CSV_File, "w") as csvfile: 
        for i in hashmap:
          hashmap[i][0] /= args.PASS
          hashmap[i][1] /= args.PASS
          writer = csv.writer(csvfile)
          writer.writerow([i,hashmap[i][0],hashmap[i][1]])
  else:
    CSV_File=args.CSV
  draw(CSV_File)
  time_end=time.time()
  print('time cost',time_end-time_start,'s')
