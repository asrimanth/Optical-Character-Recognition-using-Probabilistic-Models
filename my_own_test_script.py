from image2text import load_letters, load_training_letters, SimpleModel, HMM

def one_to_one_score(expected, actual):
    accuracy = 0
    total = len(expected)
    for i in range(total):
        if(expected[i] == actual[i]):
            accuracy += 1
    accuracy = accuracy / total
    return accuracy

if(__name__ == "__main__"):
    test_simple_model = SimpleModel()
    test_hmm_model = HMM()

    TEST_CASE_ANSWERS = ["SUPREME COURT OF THE UNITED STATES", 
    "Certiorari to the United States Court of appeals for the Sixth Circuit", 
    "Nos. 14-556. Argued April 28, 2015 - Decided June 26, 2015",
    "Together with No. 14–562, Tanco et al. v. Haslam, Governor of",
    "Tennessee, et al., also on certiorari to the same court.",
    "Opinion of the Court",
    "As some of the petitioners in these cases demonstrate, marriage",
    "embodies a love that may endure even past death.",
    "It would misunderstand these men and women to say they disrespect",
    "the idea of marriage.",
    "Their plea is that they do respect it, respect it so deeply that",
    "they seek to find its fulfillment for themselves.",
    "Their hope is not to be condemned to live in loneliness,",
    "excluded from one of civilization's oldest institutions.",
    "They ask for equal dignity in the eyes of the law.",
    "The Constitution grants them that right.",
    "The judgement of the Court of Appeals for the Sixth Circuit is reversed.",
    "It is so ordered.",
    "KENNEDY, J., delivered the opinion of the Court, in which",
    "GINSBURG, BREYER, SOTOMAYOR, and KAGAN, JJ., joined."]

    NUMBER_OF_TEST_CASES = len(TEST_CASE_ANSWERS)

    simple_mean_accuracy = 0
    hmm_mean_accuracy = 0


    for i in range(NUMBER_OF_TEST_CASES):
        test_img_fname = './test_images/test-' + str(i) + '-0.png'
        test_letters = load_letters(test_img_fname)
        train_img_fname = './test_images/courier-train.png'
        train_letters = load_training_letters(train_img_fname)
        result = test_simple_model.simple_model(train_letters, test_letters)
        hmm_result = test_hmm_model.hidden_markov_model(train_letters, test_letters)
        if(len(result) != len(TEST_CASE_ANSWERS[i])):
            print("*"*25)
            print("LENGTH MISMATCH!")
            print(f"Length in answer: {len(TEST_CASE_ANSWERS[i])}, length in result: {len(result)}")
        print("*"*25)
        hmm_result = ''.join(hmm_result)
        print(f"Answer: {TEST_CASE_ANSWERS[i]}\nSimple: {result}\n   HMM: {hmm_result}")
        simple_accuracy = one_to_one_score(TEST_CASE_ANSWERS[i], result) * 100
        hmm_accuracy = one_to_one_score(TEST_CASE_ANSWERS[i], hmm_result) * 100
        print(f"Accuracy for simple on case number {i} is: {round(simple_accuracy, 4)} %")
        print(f"Accuracy for    HMM on case number {i} is: {round(hmm_accuracy, 4)} %")
        simple_mean_accuracy += simple_accuracy
        hmm_mean_accuracy += hmm_accuracy
    
    print(f"Simple mean accuracy: {round(simple_mean_accuracy / NUMBER_OF_TEST_CASES, 4)} %")
    print(f"   HMM mean accuracy: {round(hmm_mean_accuracy / NUMBER_OF_TEST_CASES, 4)} %")