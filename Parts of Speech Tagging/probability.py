import math

def cal_prior(label):
    prior={}
    end_prior={}
    for obs in label:
        if obs[0] not in prior:
            prior[obs[0]]=1
        else:
            prior[obs[0]]+=1
        if obs[-1] not in end_prior:
            end_prior[obs[-1]]=1
        else:
            end_prior[obs[-1]]+=1
    prior={s:0.1 if s not in prior else prior[s] for s in states}
    end_prior={s:0.1 if s not in end_prior else end_prior[s] for s in states}
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
            transition[s]={s1:0.1 for s1 in states}
        else:
            transition[s]={s1:0.1 if s1 not in transition[s] else transition[s][s1]
                        for s1 in states}
    return {s1:{s:float(transition[s1][s])/denom[s1] for s in denom}
            for s1 in denom}
    
def cal_emission(data,label):
    emission={}
    denom={}
    suffix_emission={}
    suffix_denom={}
    suff_constant={'number':{s:0.9 if s=='num' else 0.00001 for s in states},\
                   'ed':{s:0.9 if s=='verb' else 0.00001 for s in states},\
                   '-like':{s:0.9 if s=='adj' else 0.00001 for s in states},\
                   'ly':{s:0.9 if s=='adv' else 0.00001 for s in states},\
                   'default_noun':{s:0.9 if s=='noun' else 0.00001 for s in states}}
	
    suffix=3
    for obs,s in zip(data,label):
        for i in range(len(obs)):
            if obs[i] not in emission:
                emission[obs[i]]={}
            else:
                if s[i] not in emission[obs[i]]:
                    emission[obs[i]][s[i]]=1
                else:
                    emission[obs[i]][s[i]]+=1
            if obs[i] not in denom:
                denom[obs[i]]=1
            else:
                denom[obs[i]]+=1    
            for suff in suff_unique:
                if obs[i].endswith(suff):
                    if suff not in suffix_emission:
                        suffix_emission[suff]={}
                    else:
                        if s[i] not in suffix_emission[suff]:
                            suffix_emission[suff][s[i]]=1
                        else:
                            suffix_emission[suff][s[i]]+=1
                    if suff not in suffix_denom:
                        suffix_denom[suff]=1
                    else:
                        suffix_denom[suff]+=1
            if len(obs[i])>suffix:
                if obs[i][-suffix:] not in suff_constant:
                    suff_constant[obs[i][-suffix:]]={}
                else:
                    if s[i] not in suff_constant[obs[i][-suffix:]]:
                        suff_constant[obs[i][-suffix:]][s[i]]=1
                    else:
                        suff_constant[obs[i][-suffix:]][s[i]]+=1
    for w in emission:
        emission[w]={s:0.1 if s not in emission[w] else emission[w][s] for s in
                     states}
    for w in suffix_emission:
        suffix_emission[w]={s:0.1 if s not in suffix_emission[w] else \
                            suffix_emission[w][s] for s in states}
    for w in suff_constant:
        suff_constant[w]={s:0.1 if s not in suff_constant[w] else \
                          suff_constant[w][s] for s in states}
    return {w:{s:float(emission[w][s])/denom[w] for s in states} for w in denom}\
           ,{w:{s:float(suffix_emission[w][s])/suffix_denom[w] for s in states} \
             for w in suffix_denom},suff_constant

def cal_prev_next_emission(data,label):
    prev_emission={}
    next_emission={}
    denom={}
    for obs,s in zip(data,label):
        for i in range(len(obs)):
            if obs[i] not in prev_emission:
                prev_emission[obs[i]]={}
            else:
                if i!=0:
                    if s[i-1] not in prev_emission[obs[i]]:
                        prev_emission[obs[i]][s[i-1]]=1
                    else:
                        prev_emission[obs[i]][s[i-1]]+=1
            if obs[i] not in next_emission:
                next_emission[obs[i]]={}
            else:
                if i!=len(obs)-1:
                    if s[i+1] not in next_emission[obs[i]]:
                        next_emission[obs[i]][s[i+1]]=1
                    else:
                        next_emission[obs[i]][s[i+1]]+=1
            if obs[i] not in denom:
                denom[obs[i]]=1
            else:
                denom[obs[i]]+=1
    for w in prev_emission:
        prev_emission[w]={s:0.1 if s not in prev_emission[w] else prev_emission[w][s]\
                          for s in states}
    for w in next_emission:
        next_emission[w]={s:0.1 if s not in next_emission[w] else next_emission[w][s]\
                          for s in states}
    return {w:{s:float(prev_emission[w][s])/denom[w] for s in states} for w in denom}\
           ,{w:{s:float(next_emission[w][s])/denom[w] for s in states} \
             for w in denom}
    
#exemplars=[]
states=['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']

suff_unique=['ing','ed','less','ful','able','ly','er','or','ar','ist','y','ous'\
             ,'al','ion','ence','ment','ness','ship','ity'\
            ,'ish','ic','ive','ate','ify','ize']
