# -*- coding: utf-8 -*-

import nltk
from nltk.corpus import brown
from nltk.corpus import conll2000
from nltk.tokenize import word_tokenize

brown = nltk.corpus.brown.tagged_words()

print """\nOblig 3A
thomavl - V13\n"""
print "Oppgave 1 - Betinget sannsynlighet"
print "(Tar litt tid)"


brown_tags = [tag for (word, tag) in brown]
bigram = nltk.bigrams(brown_tags)
cfd = nltk.ConditionalFreqDist((tag, word.lower()) for (word, tag) in brown)
cfd_bigram = nltk.ConditionalFreqDist(bigram)
cpd = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
cpd_tags = nltk.ConditionalProbDist(cfd_bigram, nltk.MLEProbDist)


print "\nOppgave 1.1.1)"
print "Det mest frekvente verbet i hele Brown-corpuset er: ", cfd['VB'].max()
print "\nOppgave 1.1.2)"
print "Time forekommer", cfd['VB']['time'], " ganger som verb (VB), og", cfd['NN']['time'], " som substantiv (NN)."
print "\nOppgave 1.1.3)"
print "Det mest frekvente adverbet er: ", cfd['RB'].max()
print "\nOppgave 1.2.1)"
print "Taggen som forekommer oftest etter et verb: ", cfd_bigram['VB'].max()
print "\nOppgave 1.2.2)"
print "Bigrammet 'DT NN' forekommer: ", cfd_bigram['DT']['NN'], " ganger."
print "\nOppgave 1.3.1)"
# valgte å bruke verbtaggen 'VB' her, fra ikke-simplified
print "Det mest sannsynlige verbet er: ", cpd['VB'].max()
print "\nOppgave 1.3.2)"
print "Sannsynligheten for substantiv etter en bestemmertagg: ", cpd_tags["DT"].prob("NN")

print "\nOppgave 2 - HMM-tagging"
print "(Les koden for kommentarer)"
# Litt usikker her om vi skulle bruke 'VBD' i starten. Sannsynligheten blir også litt vanskelig å lese siden det er så mange nuller foran ('e-08')
# printer sannsynlighet både med og uten 'VBD', svaret blir uansett sekvens[b] på begge

sannsynlighet_pps = cpd_tags["VBD"].prob("PP$") * cpd["PP$"].prob("her") * cpd_tags["PP$"].prob("NN") * cpd["NN"].prob("duck")
sannsynlighet_ppo = cpd_tags["VBD"].prob("PPO") * cpd["PPO"].prob("her") * cpd_tags["PPO"].prob("VB") * cpd["VB"].prob("duck")

# Utregningen dersom vi ikke skulle ta med 'VBD'-taggen
kun_forskjell_pps = cpd["PP$"].prob("her") * cpd_tags["NN"].prob("PP$") * cpd["NN"].prob("duck")
kun_forskjell_ppo = cpd["PPO"].prob("her") * cpd_tags["PPO"].prob("VB") * cpd["VB"].prob("duck")

print "\nDersom vi skulle ta med 'VBD'-taggen, får jeg at: "
if sannsynlighet_pps > sannsynlighet_ppo:
	print "Taggsekvens[a] (med VBD) er mest sannsynlig, med en sannsynlighet på: ", sannsynlighet_pps
else:
	print "Taggsekvens[b] (med VBD) er mest sannsynlig, med en sannsynlighet på: ", sannsynlighet_ppo

# printer ut svaret (uten 'VBD')
print "\nDersom vi ikke skulle bruke 'VBD'taggen, får jeg at: "
if kun_forskjell_pps > kun_forskjell_ppo:
	print "Taggsekvens[a] (uten VBD) er mest sannsynlig, med en sannsynlighet på: ", sannsynlighet_pps
else:
	print "Taggsekvens[b] (uten VBD) er mest sannsynlig, med en sannsynlighet på: ", sannsynlighet_pps

"""
Har ikke fått tid til å gjøre ferdig oppgave 3

print "\nOppgave 3 - Chunking"

training = conll2000.chunked_sents("train.txt", chunk_types=["NP", "PP", "VP"])
# triple quoted string
grammar = "NP: {<DT>?<JJ>*<NN.*>}
			 PP: {<IN>}
		     VP: {<VB.*><PP|CLAUSE>+$}
			 Nom: {<RB}"
cp = nltk.RegexpParser(grammar)
evaluate = nltk.chunk.util.accuracy(cp, training)
test = conll2000.chunked_sents("test.txt", chunk_types=["NP", "VP"])
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
result = cp.parse(test)
print result
print evaluate
"""
