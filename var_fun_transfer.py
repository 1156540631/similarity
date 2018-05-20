import os


f = open('wordlist.txt', encoding="utf-8")
wordlist = f.read().split('\n')
print(wordlist)

def is_identifier(s):
	if s[0] == '_' or s[0] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ': # 判断是否为字母或下划线开头
		for i in s:
			if i == '_' or i in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' or i in '0123456789': # 判断是否由字母数字或下划线组成
				pass
			else:
				return 0
		return 1
	else:
		return 0

readpath = "oldslices/"
writepath = "newslices/"
for root, _, files in os.walk(readpath):
	for name in files:
		f = open(os.path.join(root, name), encoding="utf-8")
		tmp = f.read()
		word_start_index = 0
		one_sequence = []
		counter = [1,1,1]
#these three numbers are [var,fun,class]
		for j in range(len(tmp)):
			if ((tmp[j] == ' ' or tmp[j] == '\n' or tmp[j] == '\t') and (j != word_start_index)):
				temp_word = tmp[word_start_index:j]
				word_start_index = j+1
				one_sequence.append(temp_word)
				one_sequence.append(tmp[j])
			elif ((tmp[j] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') or (tmp[j] in '0123456789')or tmp[j] == '_'):
				word_start_index = word_start_index
			elif (j == len(tmp)-1 and ((tmp[j] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') or (tmp[j] in '0123456789') or tmp[j] == '_')):
				temp_word = tmp[word_start_index:j+1]
				word_start_index = j+1
				one_sequence.append(temp_word)
			elif (tmp[j] in '!"#$%&()*+,-./:;<=>?@[\]^`{|}~'):
				if j>word_start_index:
					temp_word = tmp[word_start_index:j]
					one_sequence.append(temp_word)
				temp_word = tmp[j]
				word_start_index = j+1
				one_sequence.append(temp_word)
			else:	
				one_sequence.append(tmp[j])
				word_start_index = j+1
		def next_word(i):
			i += 1
			while (one_sequence[i] == ' ' or one_sequence[i] == '\n' or one_sequence[i] == '\t') and (i<len(one_sequence)-1):
				i += 1
			return i
		def last_word(i):
			i -= 1
			while (one_sequence[i] == ' ' or one_sequence[i] == '\n' or one_sequence[i] == '\t') and (i>0):
				i -= 1
			return i
		def replace_all_words(i,counter_index):
			target_word = one_sequence[i]
			type_of_target = ['VAR','FUN','CLASS']
			for j in range(len(one_sequence)):
				if (one_sequence[j] == target_word):
					one_sequence[j] = type_of_target[counter_index]+str(counter[counter_index])
			counter[counter_index] += 1		
		for i in range(len(one_sequence)):
			if ((one_sequence[i][0:3] != 'VAR') and (one_sequence[i][0:3] != 'FUN') and (one_sequence[i][0:5] != 'CLASS') and (is_identifier(one_sequence[i]) == 1) and (one_sequence[i] not in wordlist)):
				if ((i != len(one_sequence)-1) and (one_sequence[next_word(i)] == '*')):
					replace_all_words(i,2)
				elif ((i != 0) and (one_sequence[last_word(i)] == 'struct' or one_sequence[last_word(i)] == '}')):
					replace_all_words(i,2)
				elif (i != len(one_sequence)-1) and (one_sequence[next_word(i)] == '('):
					replace_all_words(i,1)
				elif (i>0) and (i<len(one_sequence)-1) and ((one_sequence[next_word(i)] == "'" and one_sequence[last_word(i)] == "'") or (one_sequence[last_word(i)] == '"' and one_sequence[next_word(i)] == '"')):
					pass
				elif ((i != 0) and (one_sequence[last_word(i)] == '%' or one_sequence[last_word(i)] == '\\')):
					pass
				else:
					replace_all_words(i,0)
		writefile_name = f.name.split("/")[-1]
#		print(os.getcwd())
		writefile = open(writepath+writefile_name,"w")
		write_sequence = "".join(one_sequence)
		writefile.write(write_sequence)
		writefile.close()