import math

def cal_prior(label):
    prior={}
    end_prior={}
    for obs in label:
        if obs[0] not in prior:
            prior[obs[0]]=1
        else:
            prior[obs[0]]+=1
        if obs[-4] not in end_prior:
            end_prior[obs[-4]]=1
        else:
            end_prior[obs[-4]]+=1
    prior={s:1 if s not in prior else prior[s] for s in states}
    end_prior={s:1 if s not in end_prior else end_prior[s] for s in states}
    return {s:float(prior[s])/len(label) for s in prior},\
           {s:float(end_prior[s])/len(label) for s in end_prior}

def cal_transition(label):
    transition={}
    denom={}
    for obs in label:
        for i in range(len(obs)):
            if obs[i] not in transition:
                transition[obs[i]]={}
            else:
                if i<len(obs)-1:
                    if obs[i+1] not in transition[obs[i]]:
                        transition[obs[i]][obs[i+1]]=1
                    else:
                        transition[obs[i]][obs[i+1]]+=1
            if obs[i] not in denom:
                denom[obs[i]]=1
            else:
                denom[obs[i]]+=1
    for s in states:
        if s not in transition:
            transition[s]={s1:1.0/len(states) for s1 in states}
        else:
            transition[s]={s1:1 if s1 not in transition[s] else transition[s][s1]
                        for s1 in states}
        if s not in denom:
            denom[s]=1
    return {s1:{s:float(transition[s1][s])/denom[s1] for s in denom}
            for s1 in denom}
    
def cal_emission(data):
    emission=[[0 for j in range(len(data['q'][0]))] for i in range(len(data['q']))]
    denom={}
    for s in states:
        for i in range(len(data[s])):
            for j in range(len(data[s][i])):
                if data[s][i][j]=='*':
                    emission[i][j]+=1                
    return [[float(emission[i][j])/len(states) if emission[i][j]!=0 else 1.0/len(states)\
             for j in range(len(data['q'][0]))] for i in range(len(data['q']))]

def cal_final_emission(temp_emission,train_data,test_data):
    emission={i:{} for i in range(len(test_data))}
    for k in range(len(test_data)):
        for s in states:
            prob=[temp_emission[i][j] if test_data[k][i][j]==train_data[s][i][j]\
                  else 0.00001 for i in range(len(test_data[k])) \
                  for j in range(len(test_data[k][0])) if test_data[k][i][j]=='*']
            temp=1.0
            for i in prob:
                temp*=i
            emission[k][s]=temp
    return emission

def find_emission(train_data,test_data):
    emission={i:{} for i in range(len(test_data))}
    for k in range(len(test_data)):
        for s in states:
            prob=[1 if test_data[k][i][j]==train_data[s][i][j] \
                               else 0 for i in range(len(test_data[k])) \
                               for j in range(len(test_data[k][0])) \
                                                if test_data[k][i][j]=='*']
            emission[k][s]=sum(prob)+1
    return emission
            
states=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R',\
        'S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j',\
        'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1',\
        '2','3','4','5','6','7','8','9','(',')',',','.','-','!','?','"','\'',' ']
