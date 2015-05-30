import sys
import logRegres

def main():
    dataAttr,labelsMat = logRegres.loadDataSet()
    weights = logRegres.gradAscent(dataAttr,labelsMat)
    logRegres.plotBestFit(weights)

if __name__ == "__main__":
    main()