"""dataset.py - loads the labelled news articles (label 1 = FAKE, 0 = REAL)."""
import csv
import os
import re
import sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))     # some articles are very long
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REAL_DATASET = os.path.join(DATA_DIR, "fake_or_real_news.csv")

# Source "fingerprints" that leak the label (e.g. every real ISOT article starts with "(Reuters) -").
# A model trained with them learns the tag instead of the writing, so they are removed.
LEAKS = [re.compile(r"^\s*[^()\n]{0,80}\(Reuters\)\s*[-\u2013\u2014]\s*"),
         re.compile(r"\s*(Featured image|Photo|Image) (via|by|credit)[^\n]{0,120}$", re.I),
         re.compile(r"^\s*21st Century Wire says\s*", re.I)]


def strip_leaks(text):
    for rx in LEAKS:
        text = rx.sub("", text)
    return text.strip()

FAKE = [
 "SHOCKING secret exposed! Doctors hate this one miracle fruit that cures every disease overnight, share before it gets deleted",
 "Government hiding alien base under the city, leaked insider documents prove the cover up, wake up people",
 "You won't believe what this celebrity did, the media is hiding the truth about the magic pill that melts fat in one day",
 "BREAKING: scientists confirm drinking hot water with lemon makes you immune to all viruses forever, forward to everyone now",
 "Secret society controls the weather and creates storms to punish citizens, anonymous source reveals the shocking plan",
 "Free smartphones for everyone! Click this link now and share with ten friends to claim your prize before midnight",
 "Banks will shut down all accounts tomorrow, insiders say, rush to withdraw everything before the great collapse",
 "Miracle herbal tea cures cancer in three days, big pharma furious and trying to ban this unbelievable discovery",
 "Whistleblower reveals vaccines contain tracking chips, mainstream media refuses to report the explosive truth",
 "Moon landing was filmed in a studio, new leaked video finally proves everything, they do not want you to see this",
 "Aliens spotted shopping at local market, eyewitness says the visitors left behind a glowing message for humanity",
 "Scientists shocked as man lives 200 years by eating only one secret ingredient, the elite are hiding it from you",
 "Urgent warning! Common household item causes instant death, hospitals hide the numbers, share to save your family",
 "The truth they hide: the earth is flat and satellites are fake, whistleblower engineer exposes the global lie",
 "Politician caught turning into a lizard on live camera, unbelievable footage goes viral, media silent",
 "Eat this before bed and lose twenty kilos in a week without exercise, shocking trick nutritionists do not want you to know",
 "Mobile towers are secretly mind control devices, leaked report proves the hidden agenda, spread the word fast",
 "Lottery company announces everyone who shares this post will win a million, thousands already received money",
 "Ancient prophecy says the world ends next Friday, experts confirm in a secret meeting, panic spreads everywhere",
 "Hidden cure for baldness discovered in kitchen spice, hair grows back in hours, doctors furious about this trick",
 "Exclusive: famous singer secretly replaced by a robot clone, fans stunned by the unbelievable evidence",
 "Drinking cold water after meals causes instant heart failure, shocking study the authorities want to bury",
 "Chocolate factory secretly adds mind control chemicals to sweets, brave worker leaks the horrifying recipe",
 "Scientists admit sun will be switched off for three days, stock food now, the government is lying to you",
 "This one weird trick makes your phone battery last a month, manufacturers are terrified you will find out",
 "Council quietly bans all festivals forever, sources say officials plan total control, citizens must act today",
 "Shocking video proves the ocean is shrinking by half every year, officials refuse to answer, share before removal",
 "New law forces everyone to pay tax on breathing, anonymous insider reveals the plan, the public is outraged",
 "Man discovers treasure map in old book, the museum is hiding the gold and silencing every witness",
 "Study finds that people who read this article gain superpowers in twenty four hours, doctors baffled",
]
REAL = [
 "The central bank kept its benchmark interest rate unchanged on Wednesday, citing steady inflation and moderate economic growth",
 "The state government announced a new scholarship scheme for undergraduate students, and applications will open next month on the official portal",
 "Researchers at the national university published a peer reviewed study on groundwater levels across three districts over a ten year period",
 "The health ministry reported a decline in seasonal flu cases this week and urged citizens to complete their routine vaccinations",
 "The railway department said that the new express service between the two cities will begin operations after safety inspections are completed",
 "Heavy rainfall caused waterlogging in several low lying areas on Monday, and the municipal corporation deployed pumps to clear the roads",
 "The cricket board announced the squad for the upcoming series, with two new players included after strong domestic performances",
 "Officials confirmed that the airport expansion project is on schedule, with the new terminal expected to open by the end of next year",
 "The finance ministry released quarterly data showing a modest increase in tax collection compared with the same period last year",
 "Local police arrested two suspects in connection with a theft case, and an investigation is continuing according to a department statement",
 "The education board published the examination timetable and advised students to check the official website for revised guidelines",
 "A new metro line was inaugurated on Saturday, and authorities expect it to reduce travel time for thousands of daily commuters",
 "The meteorological department forecast moderate rain over the coming week and advised farmers to plan sowing accordingly",
 "Engineers completed repairs on the damaged bridge, and traffic was restored after a structural audit by the public works department",
 "The company reported a five percent rise in quarterly revenue, driven by higher demand in its domestic markets, according to its filing",
 "Parliament passed the amendment after a three hour debate, and the bill will now be sent for the president's assent",
 "Doctors at the district hospital said that routine screening camps will be held in rural areas throughout the month",
 "The space agency successfully tested a new engine for its upcoming launch vehicle, with results to be reviewed by the technical committee",
 "Voter turnout in the municipal election was recorded at sixty one percent, the election commission said in a press note",
 "The city library extended its opening hours during the examination season to support students preparing for competitive tests",
 "A report by an independent think tank estimated that renewable energy capacity grew by eight percent over the past year",
 "The agriculture department distributed certified seeds to farmers and organised training sessions on water efficient irrigation",
 "The museum will host an exhibition of regional handicrafts starting next week, and entry will be free for school groups",
 "Authorities have issued an advisory asking residents to conserve water as reservoir levels remain below the seasonal average",
 "The university signed an agreement with a technology firm to offer internships and joint research projects for engineering students",
 "Stock markets closed slightly higher on Friday as investors reviewed corporate earnings and the latest inflation figures",
 "The transport authority introduced a digital payment system for city buses, with a pilot phase planned in two zones",
 "Wildlife officials counted a modest increase in the tiger population in the reserve during the latest annual census",
 "The hospital administration said that the new emergency wing will add fifty beds and will be operational by the spring",
 "Teachers unions met education officials on Tuesday to discuss recruitment timelines, and talks are scheduled to continue next week",
]

LABELS = {"1": 1, "FAKE": 1, "0": 0, "REAL": 0}


class DatasetLoader:
    """Provides the article texts and labels used to train the model."""
    def load(self):
        texts = FAKE + REAL
        labels = [1] * len(FAKE) + [0] * len(REAL)
        return texts, labels

    def load_csv(self, path, text_col="text", label_col="label", label=None):
        """Read articles from a CSV file. If label is given, every row gets that label
        (for corpora split into a fake file and a real file); otherwise label_col is used
        and must hold 1/0 or FAKE/REAL. Rows without text or with an unknown label are skipped."""
        texts, labels = [], []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                text = strip_leaks(row.get(text_col) or "")
                y = label if label is not None else LABELS.get(str(row.get(label_col, "")).strip().upper())
                if text and y is not None:
                    texts.append(text)
                    labels.append(y)
        return texts, labels

    def load_real(self, path=REAL_DATASET):
        """Public fake_or_real_news.csv (about 6,300 articles); run get_dataset.py first."""
        if not os.path.exists(path):
            raise FileNotFoundError("Real dataset not found. Run: python get_dataset.py")
        return self.load_csv(path)
