import pickle
import random
from tqdm import tqdm, trange
from itertools import cycle
from copy import deepcopy
import numpy as np

def gen_loss(di):
    SQUASH = 1000
    return sum([(val / SQUASH) ** 2 for val in di.values()])

def sim_add(di, gen):
    di = di.copy()
    for key in gen:
        if key in di.keys():
            di[key] -= 1/len(gen)
    return di

def randomrange(n):
    l = list(range(n))
    random.shuffle(l)
    return l

def gen_dataset(movielist, data, tv_split=0.7, v_split=0.3):
    data    = list(data.items())
    genres  = {item for sublist in data for item in sublist[1]}
    """
    gen_d   = dict(zip(genres, range(len(genres))))
    counts  = {genre: len([1 for x in data if genre in x[1]]) for genre in genres}
    """
    wcounts = {genre: sum([1/len(x[1]) for x in data if genre in x[1]]) for genre in genres}
    avg_wcount = sum(wcounts.values()) / len(wcounts.values())

    for gen, count in sorted(wcounts.items(), key=lambda x:x[1], reverse=True):
        if count >= avg_wcount * 0.1:
            pass
            #print("{:>12}: {:.2f}".format(gen, count))
        else:
            genres.remove(gen)
            #print("{:>12}: {:.2f} -- removed".format(gen, count))


    d = lambda x: {genre: wcounts[genre] * x for genre in genres}
    c = {}
    c['train'] = tv_split * (1-v_split)
    c['val']   = tv_split * v_split
    c['test']  = (1-tv_split)
    #print(c)

    state = {'train': {'gen_c': d(c['train']), 'ids': []},
               'val': {'gen_c': d(c['val']),   'ids': []},
              'test': {'gen_c': d(c['test']),  'ids': []},
              'drop': {'ids': []}}
    full_state = deepcopy(state)
    dropped = 0

    sets = ["train", "val", "test"]
    data_left = list(range(len(data)))
    while len(data_left) > 0:
        idx = random.choice(data_left)
        random.shuffle(sets)
        for s in sets:
            sim = sim_add(state[s]['gen_c'], data[idx][1])
            if len([1 for x in sim.values() if x < 0]) == 0:
                state[s]['gen_c'] = sim
                state[s]['ids']  += [idx]
                data_left.remove(idx)
                break
        else:
            state['drop']['ids'] += [idx]
            data_left.remove(idx)
            dropped += 1
    
    errs = []
    for curr, val in state.items():
        if curr == "drop": continue
        sumerr = sum(full_state[curr]['gen_c'].values())
        for gen, am in val['gen_c'].items():
            real_amount   = (full_state[curr]['gen_c'][gen] - am) / sumerr
            should_amount = wcounts[gen] / sum(wcounts.values())
            errs.append((should_amount - real_amount) / should_amount)
            val['gen_c'][gen] = errs[-1]
    
    return np.sqrt(np.sum(np.array(errs) ** 2)), state



if __name__ == '__main__':
    ml = pickle.load(open("movielist", 'rb'))
    data = {x['imdb-id']: x['genres'] for x in ml}

    best_score = np.inf
    best_state = None
    for __ in trange(100):
        score, state = gen_dataset(ml, data)
        if score < best_score:
            best_score = score
            best_state = state
    
    for key in best_state.keys():
        best_state[key]['ids'] = [ml[x]['imdb-id'] for x in best_state[key]['ids']]
        best_state[key]['labels'] = [data[x]for x in best_state[key]['ids']]
    pickle.dump(best_state, open("./sets/set_splits.p", 'wb'))

    # state2 = {s:[data[x][0] for x in v['ids']] for s, v in state.items()}
    for curr, val in best_state.items():
        if curr == "drop": continue
        fh = open("./sets/{}.csv".format(curr), 'w')
        print("==== {}-set: {} ({:.2f}%) ====".format(curr, len(val['ids']), len(val['ids']) / (len(data)) * 100))
        for gen, am in sorted(val['gen_c'].items(), key=lambda x:x[1]):
            print(" -> {:>12}: {:> 8.3f}%".format(gen, am * 100))
        for key, gens in zip(best_state[key]['ids'], best_state[key]['labels']):
            fh.write(key + "," + ",".join(gens) + "\n")
        fh.close()