import pickle
import random
from tqdm import tqdm

def gen_loss(di):
    SQUASH = 1000
    return sum([(val / SQUASH) ** 2 for val in di.values()])

def sim_add(di, gen):
    di = di.copy()
    for key in gen:
        if key in di.keys():
            di[key] -= 1/len(di)
    return di

def randomrange(n):
    l = list(range(n))
    random.shuffle(l)
    return l

def gen_dataset(movielist, tv_split=0.3, v_split=0.3):
    data    = {x['imdb-id']: x['genres'] for x in movielist}
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
            print("{:>12}: {:.2f}".format(gen, count))
        else:
            genres.remove(gen)
            print("{:>12}: {:.2f} -- removed".format(gen, count))
    
    min_wcount = min([x for x in wcounts.items() if x[0] in genres], key=lambda x:x[1])
    print(min_wcount)


    d = lambda x: dict(zip(genres, [x]*len(genres)))
    c = {}
    c['train'] = min_wcount[1] * tv_split * (1-v_split)
    c['val']   = min_wcount[1] * tv_split * v_split
    c['test']  = min_wcount[1] * (1-tv_split)

    state = {'train': {'gen_c': d(c['train']), 'ids': []},
               'val': {'gen_c': d(c['val']),   'ids': []},
              'test': {'gen_c': d(c['test']),  'ids': []}, }
    dropped = 0
    
    # for i, (id, gen) in enumerate(tqdm(data.values(), desc='-')):
    for i in tqdm(randomrange(len(data)), desc='-'):
        gen = data[i][1]
        # calculate loss for adding items
        losses = {}
        for curr, val in state.items():
            sim = sim_add(val['gen_c'], gen)
            losses[curr] = (gen_loss(val['gen_c']) - gen_loss(sim), sim)
        
        drop, (loss, sim) = max(losses.items(), key=lambda x:x[1][0])
        if loss > 0:
            state[drop]['gen_c']  = sim
            state[drop]['ids']   += [i]
            # state[drop]['img_c'] += 1
        else:
            dropped += 1
    
    pickle.dump((state, dropped), open("tmp-pickle.p", 'wb'))
        
    for curr, val in state.items():
        print("=== {}-set: {} ({:.2f}%) ===".format(curr, len(val['ids']), len(val['ids']) / (len(data) - dropped) * 100))
        for gen, am in val['gen_c'].items():
            print(" -> {}: {:> 5.1f} ({:.2f}%)".format(gen, c[curr] - am, (c[curr] - am) / min_wcount[1] * 100))

if __name__ == '__main__':
    ml = pickle.load(open("movielist", 'rb'))
    gen_dataset(ml)