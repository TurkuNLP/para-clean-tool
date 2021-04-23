from werkzeug.middleware.dispatcher import DispatcherMiddleware
import flask
from flask import Flask
from flask import render_template, request
import os
import glob
from sqlitedict import SqliteDict
import json
import datetime
import difflib
import html
import re
from collections import Counter


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
APP_ROOT = os.environ.get('PARAANN_APP_ROOT',"")
app.config["APPLICATION_ROOT"] = APP_ROOT

DATADIR=os.environ["PARAANN_DATA"]

def read_batches():
    batchdict={} #user -> batchfile -> Batch
    batchfiles=sorted(glob.glob(DATADIR+"/batches-*/archived/*.json"))
    for b in batchfiles:
        dirname,_,fname=b.split("/")[-3:]
        user=dirname.replace("batches-","")
        batchdict.setdefault(user,{})[fname]=Batch(b)
    return batchdict
    
def produce_sorted_index(batchdict):
    index = [] # (user, batchfile, idx, timestamp)
    for user in batchdict.keys():
        for batchfile in batchdict[user].keys():
            for i, pairseq in enumerate(batchdict[user][batchfile].data):
                pair = batchdict[user][batchfile].data[i]
                time = pair.get("annotation", {}).get("updated", "not updated")
                index.append((user, batchfile, i, time))
    index.sort(key=lambda x: x[-1])
    return index


class Batch:

    def __init__(self,batchfile):
        self.batchfile=batchfile
        with open(batchfile) as f:
            self.data=json.load(f) #this is a list of sentence pairs to annotate

    def save(self):
        s=json.dumps(self.data,ensure_ascii=False,indent=2,sort_keys=True)
        with open(self.batchfile,"wt") as f:
            print(s,file=f)

    @property
    def get_batch_len(self):
        batch_num = len(self.data)
        return batch_num

    @property
    def get_anno_stats(self):
        completed = 0
        skipped = 0
        left = 0
        for pair in self.data:
            if "annotation" in pair:
                if "label" in pair["annotation"]:
                    if pair["annotation"]["label"]=="x":
                        skipped += 1
                    elif "|" in pair["annotation"]["label"] or pair["annotation"]["label"].strip() == "": # label not completed
                        left+=1
                    else:
                        completed += 1
                else:
                    left += 1
            else:
                left += 1
        return (completed, skipped, left)

    @property
    def get_update_timestamp(self):
        timestamps = [datetime.datetime.fromisoformat(pair["annotation"]["updated"]) for pair in self.data if "annotation" in pair]
        if not timestamps:
            return "no updates"
        else:
            return max(timestamps).isoformat()
            
    
def get_unique_labels():
    global sorted_examples, all_batches
    labels = Counter()
    stats = {}
    for (user, batchfile, idx, time) in sorted_examples:
        pair = all_batches[user][batchfile].data[idx]
        label = pair.get("annotation", {}).get("label", "empty")
        label = norm_label(label)
        clean_status = pair.get("annotation", {}).get("clean_status", "not checked")
        labels.update([label])
        if label not in stats:
            stats[label] = Counter()
        stats[label].update([clean_status])
    return labels, stats
    
def get_unique_months():
    global sorted_examples, all_batches
    timestamps = Counter()
    stats = {}
    for (user, batchfile, idx, time) in sorted_examples:
        pair = all_batches[user][batchfile].data[idx]
        clean_status = pair.get("annotation", {}).get("clean_status", "not checked")
        if time == "not updated":
            year, month = "not", "updated"
        else:
            year, month, _ = time.split("-", 2)
        if (year, month) not in stats:
            stats[(year, month)] = Counter()
        timestamps.update([(year, month)])
        stats[(year, month)].update([clean_status])
    return timestamps, stats
    
def same_time(year, month, timestamp):
    if timestamp == "not updated":
        y, m = "not", "updated"
    else:
        y, m, _ = timestamp.split("-", 2)
    if year != y or month != m:
        return False
    return True
    
def get_next(user, batchfile, idx, timestamp, label = None, time = None):
    global all_batches, sorted_examples
    sort_idx = sorted_examples.index((user, batchfile, idx, timestamp))
    for i in range(sort_idx+1, len(sorted_examples)):
        u, b, id, t = sorted_examples[i]
        if time is not None: # return next example or None if last from this month
            if same_time(time[0], time[1], t):
                return i
            return None
        if label is not None:
            pair = all_batches[u][b].data[id]
            l = norm_label(pair.get("annotation", {}).get("label", "empty"))
            if l == label:
                return i
    return None
    
def get_prev(user, batchfile, idx, timestamp, label = None, time = None):
    global all_batches, sorted_examples
    sort_idx = sorted_examples.index((user, batchfile, idx, timestamp))
    for i in range(sort_idx-1, -1, -1):
        u, b, id, t = sorted_examples[i]
        if time is not None: # return next example or None if last from this month
            if same_time(time[0], time[1], t):
                return i
            return None
        if label is not None:
            pair = all_batches[u][b].data[id]
            l = norm_label(pair.get("annotation", {}).get("label", "empty"))
            if l == label:
                return i
    return None


def norm_label(label):
    label = "".join(label.split()) # normalize whitespace
    chars = list(label)
    for i,c in enumerate(chars):
        if c.isalnum() or c == "<" or c == ">":
            pass
        else:
            chars[i] = "-" # replace unwanted characters in order to avoid broken links
    return "".join(sorted(chars))

    
def init():
    global all_batches
    all_batches=read_batches()
    
    # in this tool original timestamps cannot be changed, so the global sorting of examples is always the same
    # produce here an index of global sorting by timestamp, later produce the "filtered" lists (example-by-timestamp, examples-by-labels) by filtering this global index and showing only relevant examples
    
    global sorted_examples
    sorted_examples = produce_sorted_index(all_batches)

init()            

@app.route('/')
def hello_world():
    global all_batches
    
    # show two categories:
    # 1) examples by label
    # 2) examples by month

    # TODO stats
    batch_stats = {} # user -> (completed, non-completed)
    for user, batches in all_batches.items():
        no_completed, no_left = 0, 0
        for batch in batches.values():
            if (batch.get_anno_stats[0] + batch.get_anno_stats[1]) == batch.get_batch_len: # completed + skipped == total
                no_completed += 1
            else:
                no_left += 1
        batch_stats[user] = (no_completed, no_left)
        
    batch_stats = {} # overwrite
    batch_stats["examples-by-month"] = (0, 0)
    batch_stats["examples-by-labels"] = (0, 0)
    return render_template("index.html",
                           app_root=APP_ROOT,
                           users=["examples-by-month", "examples-by-labels"],
                           stats=batch_stats)

@app.route("/ann/<user>")
def batchlist(user): # user is either examples-by-month or examples-by-labels
    global all_batches, sorted_examples
    
    #(done,skipped,left)
    
    
            
    batch_stats = []
    all_examples = len(sorted_examples)
    if user == "examples-by-month":
        timestamps, stats = get_unique_months()
        for t, c in timestamps.most_common(1000):
            s = stats[t]
            batch_stats.append((f"{t[0]}-{t[1]}", c, (s["OK"],s["ERROR"],s["not checked"]), "None"))
        batch_stats.sort() # sort by timestamp
    else: # examples-by-labels
        uniq_labels, stats = get_unique_labels()
        for label, c in uniq_labels.most_common(1000):
            s = stats[label]
            batch_stats.append((label, c, (s["OK"],s["ERROR"],s["not checked"]), "None"))
        
    
    return render_template("batch_list.html",app_root=APP_ROOT,batches=batch_stats,user=user,stats=(0,0,0,all_examples))#(t_done,t_skipped,t_left,total))

def prepare_pair(annotator, batchfilename, idx, pair):

    text1=pair["txt1"]
    text2=pair["txt2"]
    ann=pair.get("annotation",{})
    lab=ann.get("label","?")
    flag=ann.get("flagged", "false")
    clean_status=ann.get("clean_status", "")
    return (annotator, batchfilename, idx, ann.get("updated","not updated"),flag,lab,clean_status,text1[:50],text2[:50])


@app.route("/ann/<user>/<batchlabel>")
def jobsinbatch(user,batchlabel):
    # user: examples-by-month, examples-by-labels – batchlabel year-month, label
    global all_batches, sorted_examples
    
    pairdata = []
    
    if user == "examples-by-month":
        year, month = batchlabel.split("-")
        for (annotator, batchfile, idx, time) in sorted_examples:
            if time == "not updated":
                y, m = "not", "updated"
            else:
                y, m, _ = time.split("-", 2)
            if y!=year or m!=month:
                continue
            pair = all_batches[annotator][batchfile].data[idx]
            pairdata.append(prepare_pair(annotator, batchfile, idx, pair))

    if user == "examples-by-labels":
        for (annotator, batchfile, idx, time) in sorted_examples:
            pair = all_batches[annotator][batchfile].data[idx]
            l = pair.get("annotation", {}).get("label", "empty")
            l = norm_label(l)
            if batchlabel == l:
                pairdata.append(prepare_pair(annotator, batchfile, idx, pair))
    
    return render_template("doc_list_in_batch.html",app_root=APP_ROOT,user=user,batchfile=batchlabel,pairdata=pairdata)

@app.route("/saveann/<user>/<batchfile>/<pairseq>",methods=["POST"])
def save_document(user,batchfile,pairseq):
    global all_batches
    pairseq=int(pairseq)
    annotation=request.json
    pair=all_batches[user][batchfile].data[pairseq]
    orig_time = pair.get("annotation", {}).get("updated", "")
    orig_user = pair.get("annotation", {}).get("user", "")
    annotation["updated"]=orig_time # keep original timestamp
    annotation["user"]=orig_user # keep original annotator
    annotation["updated-cleanpara"]=datetime.datetime.now().isoformat()
    pair["annotation"]=annotation
    all_batches[user][batchfile].save()
    return "",200

@app.route("/ann/<mode>/<criteria>/<user>/<batchfile>/<pairseq>")
def fetch_document(mode, criteria, user, batchfile, pairseq):
    global all_batches, sorted_examples
    
    print(mode, criteria, user, batchfile, pairseq)
    
    pairseq=int(pairseq)
    pair=all_batches[user][batchfile].data[pairseq]
    name=pair.get("meta", {}).get("name", "").replace("\\", "").strip()
    
    # {
    # "d1": [
    #   "hs",
    #   "2020-01-08-23-00-04---84c7baba125d4592e300ffbe5e04396a.txt"
    # ],
    # "d2": [
    #   "yle",
    #   "2020-01-08-21-03-47--3-11148909.txt"
    # ],
    # "sim": 0.9922545481090577
    # }

    text1=pair["txt1"]
    text2=pair["txt2"]

    annotation=pair.get("annotation",{})
    
    if mode == "examples-by-month":
        time = criteria.split("-")
        label = None
    else:
        time = None
        label = criteria
    next_idx = get_next(user, batchfile, pairseq, annotation.get("updated", "not updated"), label=label, time=time)
    prev_idx = get_prev(user, batchfile, pairseq, annotation.get("updated", "not updated"), label=label, time=time)

    if next_idx is not None:
        next_idx = sorted_examples[next_idx]
    if prev_idx is not None:
        prev_idx = sorted_examples[prev_idx]

    return render_template("doc.html", app_root=APP_ROOT, mode=mode, criteria=criteria, text1=text1, text2=text2, pairseq=pairseq, batchfile=batchfile, user=user, annotation=annotation, name=name, next_example=next_idx, prev_example=prev_idx, is_last=(next_idx==None), is_first=(prev_idx==None))
    
    
@app.route("/ann/<user>/<batchfile>/<pairseq>") # old end point
def fetch_single_document(user, batchfile, pairseq):
    global all_batches
    
    pairseq=int(pairseq)
    pair=all_batches[user][batchfile].data[pairseq]
    name=pair.get("meta", {}).get("name", "").replace("\\", "").strip()

    text1=pair["txt1"]
    text2=pair["txt2"]

    annotation=pair.get("annotation",{})
    next_idx = None
    prev_idx = None

    return render_template("doc.html", app_root=APP_ROOT, mode="None", criteria="None", text1=text1, text2=text2, pairseq=pairseq, batchfile=batchfile, user=user, annotation=annotation, name=name, next_example=next_idx, prev_example=prev_idx, is_last=(next_idx==None), is_first=(prev_idx==None))


@app.route("/flags")
def flags():
    global all_batches
    pairdata=[]
    for user in all_batches.keys():
        for batchfile in all_batches[user].keys():
            pairs=all_batches[user][batchfile].data
            for idx,pair in enumerate(pairs):
                text1=pair["txt1"]
                text2=pair["txt2"]
                ann=pair.get("annotation",{})
                if ann:
                    lab=ann.get("label","?")
                    flag=ann.get("flagged", "false")
                    if flag=="true":
                        pairdata.append((user, batchfile, idx,ann.get("updated","not updated"),flag,lab,text1[:50],text2[:50]))
    pairdata = sorted(pairdata, key = lambda x: (x[3],x[1]), reverse=True)
    return render_template("all_flags.html",app_root=APP_ROOT,pairdata=pairdata)

@app.route("/ann/<user>/flags")
def user_flags(user):
    global all_batches
    pairdata=[]
    for batchfile in all_batches[user].keys():
        pairs=all_batches[user][batchfile].data
        for idx,pair in enumerate(pairs):
            text1=pair["txt1"]
            text2=pair["txt2"]
            ann=pair.get("annotation",{})
            if ann:
                lab=ann.get("label","?")
                flag=ann.get("flagged", "false")
                if flag=="true":
                    pairdata.append((user, batchfile, idx,ann.get("updated","not updated"),flag,lab,text1[:50],text2[:50]))
    pairdata = sorted(pairdata, key = lambda x: x[3], reverse=True)
    return render_template("user_flags.html",app_root=APP_ROOT,user=user,pairdata=pairdata)


def get_focus_region(focus, anchor):

    try: 
        _, f_span, f_line = focus.split("-")
        _, a_span, a_line = anchor.split("-")
    except: # old format
        return None, None
    if int(f_span) < int(a_span):
        return focus, anchor
    elif int(a_span) < int(f_span):
        return anchor, focus
    # span id equal
    if int(f_line) < int(a_line):
        return focus, anchor
    
    return  anchor, focus # returns min, max
        
        


    
@app.route("/ann/<user>/<batchfile>/<pairseq>/context")
def fetch_context(user,batchfile,pairseq):
    global all_batches
    pairseq=int(pairseq)
    pair=all_batches[user][batchfile].data[pairseq]

    text1=pair.get("document_context1", "")
    text2=pair.get("document_context2", "")
    
    text1=re.sub(r"\n+","\n",text1)
    text2=re.sub(r"\n+","\n",text2)

    text1=text1.replace("<i>"," ").replace("</i>"," ")
    text2=text2.replace("<i>"," ").replace("</i>"," ")

    text1=re.sub(r" +"," ",text1)
    text2=re.sub(r" +"," ",text2)

    blocks=matches(text1,text2,15) #matches are (idx1,idx2,len)
    spandata1,min1,max1=build_spans(text1,list((b[0],b[2]) for b in blocks))
    spandata2,min2,max2=build_spans(text2,list((b[1],b[2]) for b in blocks))
    
    # focus region
    left_min, left_max = get_focus_region(pair.get("focus1", None), pair.get("anchor1", None))
    right_min, right_max = get_focus_region(pair.get("focus2", None), pair.get("anchor2", None))
    
    
    return render_template("context.html", app_root=APP_ROOT, left_text=text1, right_text=text2, left_spandata=spandata1, right_spandata=spandata2, pairseq=pairseq, batchfile=batchfile, user=user, min_mlen=min(min1,min2), max_mlen=max(max1,max2)+1, mlenv=min(max(max1,max2),30), is_last=(pairseq==len(all_batches[user][batchfile].data)-1), selection_left_min=left_min, selection_left_max=left_max, selection_right_min=right_min, selection_right_max=right_max)
    
   



def matches(s1,s2,minlen=5):
    m=difflib.SequenceMatcher(None,s1,s2,autojunk=False)

    #returns list of (idx1,idx2,len) perfect matches
    return sorted(matches_r(m,s1,s2,minlen,0,len(s1),0,len(s2)), key=lambda match: (match[2], match[0]))

def matches_r(m,s1,s2,min_len,s1_beg,s1_end,s2_beg,s2_end):
    lm=m.find_longest_match(s1_beg,s1_end,s2_beg,s2_end)
    if lm.size<min_len:
        return []
    else:
        s1_left=s1_beg,lm.a
        s1_right=lm.a+lm.size,s1_end
        s1_all=(s1_beg,s1_end)
        
        s2_left=s2_beg,lm.b
        s2_right=lm.b+lm.size,s2_end
        s2_all=(s2_beg,s2_end)
        
        matches=[(lm.a,lm.b,lm.size)]
        for i1,i2 in ((s1_left,s2_left),(s1_left,s2_right),(s1_right,s2_left),(s1_right,s2_right)):
            #try all combinations of what remains
            if i1[1]-i1[0]<min_len:
                continue #too short to produce match
            if i2[1]-i2[0]<min_len:
                continue #too short to produce match
            sub=matches_r(m,s1,s2,min_len,*i1,*i2)
            matches.extend(sub)
        return matches

def build_spans(s,blocks):
    """s:string, blocks are pairs of (idx,len) of perfect matches"""
    if not blocks:
        return [], 0, 0
    #allright, this is pretty dumb alg!
    matched_indices=[0]*len(s)
    for i,l in blocks:
        for idx in range(i,i+l):
            matched_indices[idx]=max(matched_indices[idx],l)
    spandata=[]
    for c,matched_len in zip(s,matched_indices):
        #matched_len=(matched_len//5)*5
        if not spandata or spandata[-1][1]!=matched_len: #first or span with opposite match polarity -> must make new!
            spandata.append(([],matched_len))
        spandata[-1][0].append(c)
    merged_spans=[(html.escape("".join(chars)),matched_len) for chars,matched_len in spandata]
    return merged_spans, min(matched_indices),max(matched_indices) #min is actually always 0, but it's here for future need

#matches("Minulla on koira mutta sinulla on kissa.","Sinulla on kissa ja minulla on koira.")

