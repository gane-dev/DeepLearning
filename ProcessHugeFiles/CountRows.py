import csv

#
# def getstuff(filename):
#     count =0
#     with open(filename, "r",encoding="utf8") as csvfile:
#       for line in csvfile:
#         # yield next(datareader)  # yield the header row
#         count += 1
#     return count
#
# print(getstuff('../data/kg-google/train_v2.csv'))

def readFile(filename):
    count =0
    header=''
    total = 1708338
    loop = total // 17
    idx =1
    newfile = "train_"+str(count)+ ".csv"
    writefile = open('../data/kg-google/' + newfile,"w",encoding="utf8")
    with open(filename, "r",encoding="utf8") as csvfile:
        for line in csvfile:
            if count ==0:
                header = line
            if ((idx * loop) == count):
                writefile.close()
                writefile =  open('../data/kg-google/' + 'train_'+str(idx)+'.csv' ,"w",encoding="utf8")
                idx = idx+1
                writefile.writelines(header)
            writefile.writelines(line)
            count = count+1
    writefile.close()

readFile('../data/kg-google/train_v2.csv')