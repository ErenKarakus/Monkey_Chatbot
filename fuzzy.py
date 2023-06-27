import simpful as sf
import re

def fuzzy_logic():
    strength = ""
    size = ""
    power = ""

    FS = sf.FuzzySystem(show_banner=False)
    TLV = sf.AutoTriangle(3, terms=['poor', 'average', 'good'], universe_of_discourse=[0, 10])
    FS.add_linguistic_variable("size", TLV)
    FS.add_linguistic_variable("power", TLV)

    lowRating = sf.TriangleFuzzySet(0, 0, 13, term="low")
    mediumRating = sf.TriangleFuzzySet(0, 13, 25, term="medium")
    highRating = sf.TriangleFuzzySet(13, 25, 25, term="high")
    FS.add_linguistic_variable("rating", sf.LinguisticVariable([lowRating, mediumRating, highRating],
                                                               universe_of_discourse=[0, 25]))

    FS.add_rules([
        "IF (size IS poor) OR (power IS poor) THEN (rating IS low)",
        "IF (size IS average) THEN (rating IS medium)",
        "IF (size IS good) OR (power IS good) THEN (rating IS high)"
    ])
    print("Input the name of the monkey you wish to rate: ")
    strength = input("> ")
    print("Input how you would rate the size (from 0-25): ")
    size = input("> ")
    print("Input how would rate the power (from 0-25): ")
    power = input("> ")

    FS.set_variable("size", size)
    FS.set_variable("power", power)

    monkeyFuzzRating = str(FS.inference())
    monkey = [strength, size, power, monkeyFuzzRating]

    numRating = re.findall('[0-9]+', monkeyFuzzRating)
    print("Given your input, the monkey " + strength + " was rated at " + str(numRating[0]))
    return
