"""
GPU 温控：启动训练脚本，同时每隔几秒记录一次 GPU 的温度、利用率、功耗和显存，
温度达到上限时自动终止训练，保护显卡。

用法（在项目目录下运行）：
    python gpuTempGuard.py                       # 带温控运行 main.py，默认上限 80°C
    python gpuTempGuard.py --limit 75            # 把温度上限改成 75°C
    python gpuTempGuard.py --gpu 0               # 只监控 0 号 GPU（多卡服务器上使用）
    python gpuTempGuard.py -- other.py --foo 1   # 运行别的脚本：-- 后面是脚本和它的参数

训练输出会同时显示在屏幕上并保存到 logs/train-时间.log，GPU 记录保存到 logs/gpu-时间.csv。
需要 NVIDIA 显卡和 nvidia-smi（安装 NVIDIA 驱动后自带）。
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

projectDir = os.path.dirname(os.path.abspath(__file__))  # 项目目录，训练脚本在这里运行
statusEvery = 30      # 每隔多少秒在屏幕上打印一次 GPU 状态
cooldownSeconds = 10  # 训练结束后继续记录多少秒，观察降温情况


def parseArgs():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='带 GPU 温度保护地运行训练脚本')
    parser.add_argument('--limit', type=float, default=80, help='温度上限（°C），达到即终止训练，默认 80')
    parser.add_argument('--interval', type=float, default=2, help='采样间隔（秒），默认 2')
    parser.add_argument('--gpu', type=int, default=None, help='只监控指定编号的 GPU，默认监控全部')
    parser.add_argument('command', nargs='*', default=['main.py'], help='要运行的脚本及其参数，默认 main.py')
    return parser.parse_args()


def toNumber(text):
    """把 nvidia-smi 输出的字段转成数字；显卡不支持的字段（显示为 [N/A]）记为 NaN"""
    try:
        return float(text.strip())
    except ValueError:
        return math.nan


def readGpuStats(smiPath, gpuIndex):
    """调用 nvidia-smi 读取 GPU 状态。返回列表，每块 GPU 一项：[编号, 温度, 利用率, 功耗, 显存]"""
    command = [smiPath, '--query-gpu=index,temperature.gpu,utilization.gpu,power.draw,memory.used',
               '--format=csv,noheader,nounits']
    if gpuIndex is not None:
        command += ['-i', str(gpuIndex)]
    output = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True).stdout
    return [[toNumber(value) for value in line.split(',')] for line in output.strip().splitlines()]


def forwardOutput(stream, logFile):
    """把训练进程的输出逐行打印到屏幕，同时写入日志文件（在单独的线程里运行）"""
    for line in stream:
        sys.stdout.write(line)
        sys.stdout.flush()
        logFile.write(line)
        logFile.flush()


def stopProcess(process):
    """终止训练进程：先请求退出，5 秒内没退出就强制结束"""
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def describe(values):
    """返回 (最大值, 平均值)，忽略 NaN；全是 NaN 时返回 (NaN, NaN)"""
    validValues = [value for value in values if not math.isnan(value)]
    if not validValues:
        return math.nan, math.nan
    return max(validValues), sum(validValues) / len(validValues)


def writeCsv(csvPath, records):
    """把所有采样记录写成 CSV，方便用 Excel 打开或画图"""
    with open(csvPath, 'w', encoding='utf-8') as csvFile:
        csvFile.write('seconds,phase,gpu,temperatureC,utilizationPct,powerW,memoryMiB\n')
        for record in records:
            csvFile.write(','.join(str(value) for value in record) + '\n')


def printSummary(records):
    """按 GPU 打印训练期间的温度、利用率、功耗、显存汇总"""
    for gpuIndex in sorted({record[2] for record in records}):
        trainRecords = [r for r in records if r[2] == gpuIndex and r[1] == 'train']
        if not trainRecords:
            continue
        maxTemperature, avgTemperature = describe([r[3] for r in trainRecords])
        maxUtilization, avgUtilization = describe([r[4] for r in trainRecords])
        maxPower, avgPower = describe([r[5] for r in trainRecords])
        maxMemory, _ = describe([r[6] for r in trainRecords])
        print(f'[温控] GPU {gpuIndex}：温度 最高 {maxTemperature:.0f}°C / 平均 {avgTemperature:.1f}°C；'
              f'利用率 最高 {maxUtilization:.0f}% / 平均 {avgUtilization:.0f}%；'
              f'功耗 最高 {maxPower:.0f}W / 平均 {avgPower:.0f}W；显存最高 {maxMemory:.0f}MiB')
        cooldownRecords = [r for r in records if r[2] == gpuIndex and r[1] == 'cooldown']
        if cooldownRecords:
            print(f'[温控] GPU {gpuIndex}：训练结束 {cooldownSeconds} 秒后温度 {cooldownRecords[-1][3]:.0f}°C')


def main():
    args = parseArgs()
    sys.stdout.reconfigure(errors='replace')  # 防止个别控制台无法显示某些字符时报错

    smiPath = shutil.which('nvidia-smi')
    if smiPath is None:
        sys.exit('[温控] 找不到 nvidia-smi：需要 NVIDIA 显卡和驱动才能监控温度')

    # 日志文件放在 logs/ 目录，用启动时间命名，多次运行不会互相覆盖
    logDir = os.path.join(projectDir, 'logs')
    os.makedirs(logDir, exist_ok=True)
    timeStamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    trainLogPath = os.path.join(logDir, f'train-{timeStamp}.log')
    csvPath = os.path.join(logDir, f'gpu-{timeStamp}.csv')

    records = []  # 每条记录：(秒数, 阶段, GPU 编号, 温度, 利用率, 功耗, 显存)
    startTime = time.time()

    def sample(phase):
        """采样一次所有被监控 GPU 的状态并记录，返回其中的最高温度"""
        seconds = round(time.time() - startTime, 1)
        gpuStats = readGpuStats(smiPath, args.gpu)
        for gpuIndex, temperature, utilization, power, memory in gpuStats:
            records.append((seconds, phase, int(gpuIndex), temperature, utilization, power, memory))
        return describe([stat[1] for stat in gpuStats])[0]

    idleTemperature = sample('idle')
    if math.isnan(idleTemperature):
        sys.exit('[温控] 这块显卡不支持读取温度，无法进行温控')
    print(f'[温控] 训练前 GPU 温度 {idleTemperature:.0f}°C，上限 {args.limit:.0f}°C，'
          f'每 {args.interval:g} 秒采样一次', flush=True)

    # 启动训练子进程：-u 表示不缓冲输出，训练日志能实时显示
    process = subprocess.Popen([sys.executable, '-u', *args.command], cwd=projectDir,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace',
                               env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    logFile = open(trainLogPath, 'w', encoding='utf-8')
    readerThread = threading.Thread(target=forwardOutput, args=(process.stdout, logFile), daemon=True)
    readerThread.start()

    stoppedByGuard = False
    lastStatusTime = time.time()
    try:
        while process.poll() is None:
            currentTemperature = sample('train')
            if currentTemperature >= args.limit:
                print(f'[温控] GPU 温度 {currentTemperature:.0f}°C 达到上限 {args.limit:.0f}°C，终止训练！', flush=True)
                stopProcess(process)
                stoppedByGuard = True
                break
            if time.time() - lastStatusTime >= statusEvery:
                print(f'[温控] 当前 GPU 温度 {currentTemperature:.0f}°C', flush=True)
                lastStatusTime = time.time()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('[温控] 收到 Ctrl+C，终止训练', flush=True)
        stopProcess(process)
    except Exception as error:  # 例如 nvidia-smi 读取失败：无法监控温度时，为安全起见终止训练
        print(f'[温控] 读取 GPU 状态失败（{error}），为安全起见终止训练', flush=True)
        stopProcess(process)
        stoppedByGuard = True
    finally:
        process.wait()
        readerThread.join(timeout=5)
        logFile.close()

    # 训练结束后再记录一会儿，看显卡降温情况（按 Ctrl+C 可跳过）
    try:
        for _ in range(max(1, round(cooldownSeconds / args.interval))):
            time.sleep(args.interval)
            sample('cooldown')
    except (KeyboardInterrupt, Exception):
        pass
    writeCsv(csvPath, records)

    print('[温控] ========== 汇总 ==========')
    print(f'[温控] 训练进程退出码 {process.returncode}' + ('（被温控终止）' if stoppedByGuard else ''))
    printSummary(records)
    print(f'[温控] 训练日志：{trainLogPath}')
    print(f'[温控] GPU 记录：{csvPath}')
    sys.exit(1 if stoppedByGuard else process.returncode)


if __name__ == '__main__':
    main()
