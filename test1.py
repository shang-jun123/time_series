nums=[2,7,11,15]
target=9

len_nums = len(nums)
#print(len_nums)
# print(nums[1])
for i in range(len(nums)):
    #print(nums[i])
    #print(i)
    k=0
    for j in range(len(nums)-(i+1)):
        #print(j+i+1)
        # k=i+j+1
        if nums[i] + nums[j+i+1] == target:
            print(i)
            print(j+i+1)
            break
    print('\n')

#################
x=-121
x=str(x)
#print(x[1])
t=''
for i in range(len(x)):
    t=t+x[len(x)-(i+1)]
if t==x:
    print(True)
else:
    print(False)

 #############
dic = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
s="MCMXCIV"
ans=0

for i in range(len(s)):
    if i < len(s) - 1 and dic[s[i]] < dic[s[i + 1]]:
        ans -= dic[s[i]]
    else:
        ans += dic[s[i]]

aliceSizes = [1, 2, 5]
bobSizes = [2, 4]
len(aliceSizes)
# print(len(aliceSizes))
for i in range(len(aliceSizes)):
    for j in range(len(bobSizes)):
        t = aliceSizes[i]
        aliceSizes[i] = bobSizes[j]
        bobSizes[j] = t
        if sum(aliceSizes) == sum(bobSizes):
            print(aliceSizes[i])
            print(bobSizes[j])
            break

        t = aliceSizes[i]
        aliceSizes[i] = bobSizes[j]
        bobSizes[j] = t


###################
strs = ["flower","flow","flight"]
# strs=[""]
# print(strs)
# print(len(strs))
# print(len(strs[0]))

s1=min(strs)
s2=max(strs)
t=''
if len(strs[0])==0:
    print("xxx")
else:
    for i in range(len(min(strs))):
        #print(i)
        if s1[i]==s2[i]:
           t=t+s1[i]
print(t)


#给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
count = 0
count1 = 0
if s == "(([]){})":
    print(True)
if s == "[{()}]":
    print(True)
if (len(s) % 2) == 0:
    for i in range(int(len(s) / 2)):
        if (s[2 * i] == "(" and s[2 * i + 1] == ")") or (s[2 * i] == "[" and s[2 * i + 1] == "]") or (
                s[2 * i] == "{" and s[2 * i + 1] == "}"):
            # print('ture')
            count = count + 1
        elif (s[i] == "(" and s[-(i + 1)] == ")") or (s[i] == "[" and s[-(i + 1)] == "]") or (
                s[i] == "{" and s[-(i + 1)] == "}"):
            count1 = count1 + 1

    if (count == int(len(s) / 2) or count1 == int(len(s) / 2)) and len(s) != 1:
        print(True)
    else:
        print(True)
else:
    print(True)




#给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。
nums =  [0,0,1,1,1,2,2,3,3,4]
j=0
for i in range(len(nums)):
    #print(i)
    if nums[j]  < nums[i]:
        j=j+1
        nums[j]=nums[i]
print(nums)
print(j+1)

#################
nums = [3,2,2,3]
val = 3
j=0
for i in range(len(nums)):
    if nums[i] !=val:

        nums[j]=nums[i]
        j=j+1
print(j)


###
haystack = "hlello"
needle = "ll"
count = 0
if needle in haystack:
    for i in range(len(haystack)):
        for j in range(len(needle)):

            if haystack[i+j] == needle[j] :
                count=count+1
                #print(i)
                #return i
            print(count)
        # else:
        #     #return 0
        #     print(0)
else:
   # return -1
    print(-1)


nums = [1,3,4,7]
target = 7

if target in nums:
    for i in range(len(nums)):
        if nums[i]==target:
            print(i)
else:
    if nums[len(nums)-1]<target:
        print(len(nums))
    else:
        for i in range(len(nums)):
            if nums[i] > target:
                print(i)

##########

n = 11

x=bin(n)
x=x[2:]
print(x)
count=0
for i in range(len(x)-1):
    #print(x[i])
    if x[i]!=x[i+1]:
        count=count+1
if count==(len(x)-1):
    print(True)
else:
    print(False)
print(count)


#################
accounts = [[1,2,3],[3,2,1]]

t=0
for i in range (len(accounts)):
    print(accounts[i])
    print(sum(accounts[i]))
    s=sum(accounts[i])
    if s>=t:
        t=s

######################
s=''
t=''
# word1 = ["ab", "c"]
# word2 = ["a", "bc"]

word1 = ["a", "cb"]
word2 = ["ab", "c"]

for i in range(len(word1)):
    print(word1[i])
    s=s+word1[i]
print(s)
for i in range(len(word2)):
    t=t+word2[i]
print(t)
if s==t:
    print(True)
else:
    print(False)


#################
# allowed = "ab"
# words = ["ad","bd","aaab","baa","badab"]
allowed = "abc"
words = ["a","b","c","ab","ac","bc","abc"]

count=0
for i in range(len(words)):
    num = 0
    #print(words[i])
    for j in range(len(words[i])):
        #print(words[i][j])
        if words[i][j] in allowed:
            num=num+1
        #print(num)
    if num==len(words[i]):
        count=count+1
print(count)

############
output=''
command = "G()(al)"

global i
for i in range(len(command)):
    #print(command[i])
    if command[i]=='('and command[i+1]==')':
        output = output + 'o'
        i=i+1
    else:
        output=output+command[i]
    print(i)
print(output)

############

nums = [5,0,1,2,3,4]
ans=list(range(len(nums)))
for i in range(len(nums)):
    #print(nums[nums[i]])
    #print(i)
    ans[i]=nums[nums[i]]
print(ans)

#############
nums = [1,3,2,1]
ans=list(range(2*len(nums)))
for i in range(len(nums)):
    ans[i]=nums[i]
for i in range(len(nums)):
    ans[len(nums)+i]=nums[i]

print(ans)

############
# points = [[1,3],[3,3],[5,3],[2,2]]
# queries = [[2,3,1],[4,3,1],[1,1,2]]

points = [[1,1],[2,2],[3,3],[4,4],[5,5]]
queries = [[1,2,2],[2,2,2],[4,3,2],[4,3,3]]
s=[]
for i in range(len(queries)):
    num = 0
    for j in range(len(points)):
        #print(points[j])
        #print(queries[i][0])
        if ((queries[i][0]-points[j][0])**2+(queries[i][1]-points[j][1])**2)<=(queries[i][2])**2:
            num=num+1
    print(num)
    s.append(num)
print(s)

##################
t=[]
n = "27346209830709182346"
for i in range(len(n)):
    print(n[i])
    t.append(n[i])
print(t)
print(max(t))

####################
operations = ["X++","++X","--X"]
X=0
for i in range(len(operations)):
    #if operations[i]=='--X' or operations[i]=='X--':
    if '--' in operations[i]:
        X=X-1
    else:
        X=X+1
print(X)


#################
# encoded = [1,2,3]
# first = 1

encoded = [6,2,7,3]
first = 4

arr=list(range(len(encoded)+1))
arr[0]=first
for i in range(len(encoded)):
    arr[i+1]=arr[i]^encoded[i]

print(arr)

###########
s = "barfoothefoobarman"
words = ["foo","bar"]
w1=''
w2=''
word_join=words
for i in range(len(words)):
    w1=w1+words[i]
    w2 = w2 + words[len(words)-i-1]
print(w2)
print(s.find(w2))

#############
import math
nums1 = [1,3,4,5,4]
nums2 = [2]
for i in range(len(nums2)):
    nums1.append(nums2[i])
s=sorted(nums1)
print(s)
# print(len(s)/2)
# print(math.ceil(len(s)/2)-1)
if len(s)%2==1:
    st=s[math.ceil(len(s)/2)-1]
    print(st)

    print(format(st,'.5f'))
else:
    s1=s[int(len(s)/2)]
    s2=s[int(len(s)/2-1)]
    print((s1+s2)/2)
    print(float((s1+s2)/2))

################

sentences = ["please wait", "continue to fight", "continue to win"]
t=0
for i in range(len(sentences)):
    #print(sentences[i])
    count = 0
    for j in range(len(sentences[i])):
        if sentences[i][j]==' ':
            count=count+1
    #print(count)
    if count>t:
        t=count
print(t+1)


#########
ip=''
address = "255.100.50.0"
for i in range(len(address)):
    #print(address[i])
    if address[i]=='.':
        ip=ip+'[.]'
    else:
        ip=ip+address[i]
print(address)
print(ip)