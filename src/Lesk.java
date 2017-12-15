/**
 * Implement the Lesk algorithm for Word Sense Disambiguation (WSD)
 */
import java.util.*;
import java.io.*;
import javafx.util.Pair;

import edu.mit.jwi.*;
import edu.mit.jwi.Dictionary;
import edu.mit.jwi.item.*; 
import edu.mit.jwi.item.POS;
import edu.mit.jwi.data.parse.SenseKeyParser;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.logging.RedwoodConfiguration;

//import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.pipeline.*;
//import java.util.*;

class SenseScore {
	public String senseKey;
	public int frequency;
	public double simScore;

	public SenseScore(String senseKey, int frequency, double simScore){
		this.senseKey = senseKey;
		this.frequency = frequency;
		this.simScore = simScore;
	}

	@Override
	public String toString(){
		return senseKey + " : frequency = " + frequency + " SimScore = " + simScore;
	}

	public static int compare(SenseScore a, SenseScore b){
		int simComp = Double.compare(a.simScore, b.simScore);
		if(simComp == 0){
			return Integer.compare(a.frequency, b.frequency);
		}
		return simComp;
	}
}

public class Lesk {

	/** 
	 * Each entry is a sentence where there is at least a word to be disambiguate.
	 * E.g.,
	 * 		testCorpus.get(0) is Sentence object representing
	 * 			"It is a full scale, small, but efficient house that can become a year' round retreat complete in every detail."
	 **/
	private ArrayList<Sentence> testCorpus = new ArrayList<Sentence>();
	
	/** Each entry is a list of locations (integers) where a word needs to be disambiguate.
	 * The index here is in accordance to testCorpus.
	 * E.g.,
	 * 		ambiguousLocations.get(0) is a list [13]
	 * 		ambiguousLocations.get(1) is a list [10, 28]
	 **/
	private ArrayList<ArrayList<Integer> > ambiguousLocations = new ArrayList<ArrayList<Integer> >();
	
	/**
	 * Each entry is a list of pairs, where each pair is the lemma and POS tag of an ambiguous word.
	 * E.g.,
	 * 		ambiguousWords.get(0) is [(become, VERB)]
	 * 		ambiguousWords.get(1) is [(take, VERB), (apply, VERB)]
	 */
	private ArrayList<ArrayList<Pair<String, String> > > ambiguousWords = new ArrayList<ArrayList<Pair<String, String> > > (); 
	
	/**
	 * Each entry is a list of maps, each of which maps from a sense key to similarity(context, signature)
	 * E.g.,
	 * 		predictions.get(1) = [{take%2:30:01:: -> 0.9, take%2:38:09:: -> 0.1}, {apply%2:40:00:: -> 0.1}]
	 */
	private ArrayList<ArrayList<ArrayList<SenseScore>>> predictions = new ArrayList<ArrayList<ArrayList<SenseScore> > >();
	
	/**
	 * Each entry is a list of ground truth senses for the ambiguous locations.
	 * Each String object can contain multiple synset ids, separated by comma.
	 * E.g.,
	 * 		groundTruths.get(0) is a list of strings ["become%2:30:00::,become%2:42:01::"]
	 * 		groundTruths.get(1) is a list of strings ["take%2:30:01::,take%2:38:09::,take%2:38:10::,take%2:38:11::,take%2:42:10::", "apply%2:40:00::"]
	 */
	private ArrayList<ArrayList<String> > groundTruths = new ArrayList<ArrayList<String> >();
	
	/* This section contains the NLP tools */
	
	private static final Set<String> POSstrings = new HashSet<String>(Arrays.asList("ADJECTIVE", "ADVERB", "NOUN", "VERB"));
	
	public static final String ALL_WORDS = "ALL_WORDS";
	public static final String ALL_WORDS_R = "ALL_WORDS_R";
	public static final String WINDOW = "WINDOW";
	public static final String _POS = "POS";

	private static IDictionary wordnetdict;
	
	private static StanfordCoreNLP pipeline;

	private static Set<String> stopwords;
	
	private static IDictionary initDictionary(){
		String path = "/home/billy/Documents/nlp/p4_nlp/data/dict";
		File dictFile = null;
		dictFile = new File(path);
		if(dictFile == null) return null;

		IDictionary dict = new Dictionary(dictFile);
		try{ 
			dict.open();
		} 
		catch(IOException e){
			e.printStackTrace();
			return null; 
		}

		return dict;
	}

	private static Set<String> readStopWords(){
		String path = "/home/billy/Documents/nlp/p4_nlp/data/stopwords.txt";
		Set<String> out = null;

		//Based on gtonic and Knobo answer on stack overflow
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));
			String line = br.readLine();
			out = new HashSet<String>();
			while (line != null) {
				out.add(line);
				line = br.readLine();
			}
			br.close();
		}catch(FileNotFoundException fnfe){
			fnfe.printStackTrace();
			return null;
		}catch(IOException ioe){
			ioe.printStackTrace();
			return null;
		}

		return out;
	}

	private static StanfordCoreNLP initPipeline(){
		// Create StanfordCoreNLP object properties, with POS tagging
        // (required for lemmatization), and lemmatization
        Properties props;
        props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma");

        // StanfordCoreNLP loads a lot of models, so you probably
        // only want to do this once per execution
		//pipeline = new StanfordCoreNLP(props);
		return new StanfordCoreNLP(props);
	}

	/**
	 * TODO:
	 * The constructor initializes any WordNet/NLP tools and reads the stopwords.
	 */
	public Lesk() {
		//Dictionary, stopwords, and the NLP pipeline are already done statically
		//String[] x = {"hello", "goodbye"};
		//String[] y = {"hello", "kitty"};
		// ArrayList<String> bag1 = new ArrayList<String>();
		// bag1.add("hello");
		// bag1.add("goodbye");
		// ArrayList<String> bag2 = new ArrayList<String>();
		// bag2.add("hello");
		// bag2.add("kitty");
		// double sim = similarity(bag1, bag2, "JACCARD");
		// System.out.println(sim);
		// System.exit(1);
	}
	
	/**
	 * Convert a pos tag in the input file to a POS tag that WordNet can recognize (JWI needs this).
	 * We only handle adjectives, adverbs, nouns and verbs.
	 * @param pos: a POS tag from an input file.
	 * @return JWI POS tag.
	 */
	private String toJwiPOS(String pos) {
		if (pos.equals("ADJ")) {
			return "ADJECTIVE";
		} else if (pos.equals("ADV")) {
			return "ADVERB";
		} 
		// else if (pos.equals("CONJ")) { //Billy added
		// 	return "ADVERB";
		// } else if (pos.equals("PRON")) { //Billy added
		// 	return "NOUN";
		// } else if (pos.equals("ADP")) { //Billy added
		// 	return "ADJECTIVE";
		// } else if (pos.equals("CONJ")) { //Billy added
		// 	return "ADVERB";
		// } 
		else if (pos.equals("NOUN") || pos.equals("VERB")) {
			return pos;
		} else {
			return null;
		}
	}

	private void inputSentence(String line){
		Annotation annotation = new Annotation(line);
		this.pipeline.annotate(annotation);
		Sentence mySentence = new Sentence();

		//from cricket_007 and chris on stack overflow

		// Iterate over all of the sentences found
		List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
		//if(sentences.size() > 1) System.out.println("Sentence split: " + line);
        for(CoreMap sentence: sentences) {
            // Iterate over all tokens in a sentence
            for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
                // Retrieve and add the lemma for each word into the list of lemmas
				//lemmas.add(token.get(LemmaAnnotation.class));
				String lemme = token.lemma(); //get(LemmaAnnotation.class);
				String posTag = token.tag();
				Word w = new Word(lemme, posTag);
				mySentence.addWord(w);
            }
        }
		//System.out.println(mySentence.toString());
		testCorpus.add(mySentence);
	}

	private void processSense(String line){
		String[] words = line.split(" ");
		int index = -1;
		try{
			index = Integer.parseInt(words[0]);
		}catch(NumberFormatException nfe){
			index = -1;
			return;
		}
		if(toJwiPOS(words[2]) == null) return; //skip not adj, noun, verbs


		ambiguousLocations.get(sentenceIndex).add(index);
		if(words.length < 4){
			System.out.println("Not enough sense information: " + line);
			return;
		}
		Sentence x = testCorpus.get(sentenceIndex);
		String lemma = words[1];
		try{
			lemma = x.getWordAt(index).getLemme();
		}catch(NullPointerException npe){
			System.out.println(x.toString());
			System.out.println("Sentence #: " + sentenceIndex + " word Index: " + index);
			npe.printStackTrace();
		}
		
		Pair<String, String> wordPOS = new Pair<String,String>(lemma, words[2]);
		ambiguousWords.get(sentenceIndex).add(wordPOS);

		groundTruths.get(sentenceIndex).add(words[3]);
	}

	private int sentenceIndex;
	private boolean newSentence;

	private void processLine(String line){
		if(line.charAt(0) == '#'){
			//The line is a sense definition of a particular ambiguous word
			processSense(line.substring(1));
			newSentence = false;
		}else{
			String[] words = line.split(" ");
			int numAmbiguous = -1;
			try{
				numAmbiguous = Integer.parseInt(words[0]);
			}catch(NumberFormatException nfe){
				numAmbiguous = -1;
			}
			if(numAmbiguous < 0 || words.length > 1 ){
				//The line is a new sentence
				inputSentence(line);
				newSentence = true;
				sentenceIndex++;
				ambiguousWords.add(new ArrayList<Pair<String, String>>());
				//System.out.println("New Sentence");
			}else{
				//The line is the number of the ambiguous words in the sentence
				if(newSentence){
					ambiguousLocations.add(new ArrayList<Integer>(numAmbiguous));
					groundTruths.add(new ArrayList<String>(numAmbiguous));
				}
				newSentence = false;
			}
		}
	}

	/**
	 * TODO:
	 * Read in sentences and ambiguous words in a test corpus to fill the data structures testCorpus,
	 * locations, targets and ground_truths. The format of an input file is specified above.
	 * During testing, I will feed different test corpus files to this function.
	 * 
	 * This function fills up testCorpus, ambiguousLocations and groundTruths lists
	 * @param filename
	 */
	public void readTestData(String filename) throws Exception {
		sentenceIndex = -1;
		newSentence = false;
		ambiguousLocations.add(new ArrayList<Integer>());
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line = br.readLine();
			//System.out.println(line);
			while (line != null) {
				processLine(line);
				line = br.readLine();
			}
			br.close();
		}catch(FileNotFoundException fnfe){
			fnfe.printStackTrace();
		}catch(IOException ioe){
			ioe.printStackTrace();
		}
	}
	
	/**
	 * TODO:
	 * For a particular combination of lemma (a wordform to be exact) and a POS tag, query
	 * WordNet to find all senses and corresponding glosses.
	 * 
	 * Create signatures of the senses of a pos-tagged word.
	 * 
	 * 1. use lemma and pos to look up IIndexWord using Dictionary.getIndexWord()
	 * 2. use IIndexWord.getWordIDs() to find a list of word ids pertaining to this (lemma, pos) combination.
	 * 3. Each word id identifies a sense/synset in WordNet: use Dictionary's getWord() to find IWord
	 * 4. Use the getSynset() api of IWord to find ISynset
	 *    Use the getSenseKey() api of IWord to find ISenseKey (such as charge%1:04:00::)
	 * 5. Use the getGloss() api of the ISynset interface to get the gloss String
	 * 6. Use the Dictionary.getSenseEntry(ISenseKey).getTagCount() to find the frequencies of the synset.d
	 * 
	 * @param args
	 * lemma: word form to be disambiguated
	 * pos_name: POS tag of the wordform, must be in {ADJECTIVE, ADVERB, NOUN, VERB}.
	 * 
	 */
	private Map<String, Pair<String, Integer> > getSignatures(String lemma, String pos_name) {
		IIndexWord idxWord;
		try{
			idxWord = wordnetdict.getIndexWord(lemma, POS.valueOf(toJwiPOS(pos_name)));
		}catch(Exception e){
			System.out.println("Lemma: " + lemma + " POS: " + pos_name);
			e.printStackTrace();
			return null;
		}
		
		List<IWord> senses = new ArrayList<IWord>();
		Map<String, Pair<String, Integer> > signatures = new HashMap<String, Pair<String, Integer> >();

		if (idxWord != null)
		{
			for (IWordID senseID : idxWord.getWordIDs()){
				senses.add(wordnetdict.getWord(senseID));

				IWord word = wordnetdict.getWord(senseID);
				ISynset synset = word.getSynset();
				ISenseKey senseKey = word.getSenseKey();
				String gloss = synset.getGloss();
				Integer frequency = wordnetdict.getSenseEntry(senseKey).getTagCount();
				Pair<String, Integer> signature = new Pair<String,Integer>(gloss, frequency);
				signatures.put(senseKey.toString(), signature);
				//System.out.println("word: " + word.toString() + " synset: " + synset.toString() + " senseKey: " + senseKey.toString());
			}
		}
		
		return signatures;
	}
	
	private ArrayList<String> removeStopWords(ArrayList<String> original){
		int max = original.size();
		for(int i = 0; i < max ; i++){
			if(stopwords.contains(original.get(i))){
				original.remove(i);
				i--;
				max--;
			}
		}
		return original;
	}

	/**
	 * TODO:
	 * Convert a String object, usually representing one or more sentences, to a bag-of-words.
	 * You may want to do tokenization, lemmatization, lower-casing here and leave the stopword
	 * removal to the predict function.
	 * 
	 * Create a bag-of-words representation of a document (a sentence/phrase/paragraph/etc.)
	 * @param str: input string
	 * @return a list of strings (words, punctuation, etc.)
	 */
	private ArrayList<String> str2bow(String str, String context_option) {
		ArrayList<String> bow = new ArrayList<String>();
		Annotation annotation = new Annotation(str);
		this.pipeline.annotate(annotation);


		//from cricket_007 and chris on stack overflow

		// Iterate over all of the sentences found
		List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);

		for (CoreMap sentence : sentences) {
			// Iterate over all tokens in a sentence
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				// Retrieve and add the lemma for each word into the list of lemmas
				String lemme = token.lemma();
				lemme = lemme.toLowerCase();
				// String posTag = token.tag();
				// Word w = new Word(lemme, posTag);
				String alphanumaric = "^[a-zA-Z0-9]*$";
				if(lemme.matches(alphanumaric))
					bow.add(lemme);
			}
		}
		if(context_option.equals(ALL_WORDS_R)){
			bow = removeStopWords(bow);
		}
		//System.out.println(bow.toString());
		return bow;
	}
	
	private ArrayList<String> getContext(Sentence sentence, String context_option, int window_size, int targetIndex) {
		ArrayList<String> bow = sentence.getAllWords();
		int max = bow.size();
		for(int i = 0; i < max; i++){
			String word = bow.get(i);
			String alphanumaric = "^[a-zA-Z0-9]*$";
			if(word.matches(alphanumaric)){
				bow.remove(i);
				bow.add(i, word.toLowerCase());
			}else{
				bow.remove(i);
				i--;
				max--;
			}	
		}
		if(context_option.equals(ALL_WORDS_R)){
			bow = removeStopWords(bow);
		}else if(context_option.equals(WINDOW) && targetIndex >=0 && targetIndex < bow.size()){
			int radius = (window_size - 1) / 2;
			int low = targetIndex - radius;
			low = low < 0 ? 0 : low;
			int high = targetIndex + radius + 1; //Accounts for exclusive subArray
			high = high > bow.size() ? bow.size(): high;
			bow = new ArrayList<String>(bow.subList(low, high));
		}else if(context_option.equals(_POS)){
			//Not implemented
		}
		//System.out.println(bow.toString());
		return bow;
	}

	private double jaccardSimilarity(ArrayList<String> bag1, ArrayList<String> bag2) {
		int intersection = 0;
		HashSet<String> contains = new HashSet<String>(bag2);
		//System.out.println(bag1.toString());
		//System.out.println(contains.toString());
		for (String word : bag1) {
			if (contains.contains(word)) {
				intersection++;
				//System.out.println("Intersection = " + intersection);
			} else {
				//System.out.println(word + " != " + bag2.get(0) + " || " + bag2.get(1));
			}
		}
		int union = bag1.size() + bag2.size() - intersection;
		return intersection / (double) union;
	}

	private double cosineSimilarity(ArrayList<String> bag1, ArrayList<String> bag2) {
		int dotproduct = 0;
		HashSet<String> contains = new HashSet<String>(bag2);
		for (String word : bag1) {
			if (contains.contains(word)) {
				dotproduct++;
			}
		}
		double magB1 = Math.sqrt((double)bag1.size());
		double magB2 = Math.sqrt((double)bag2.size());
		return (double)dotproduct / (magB1 * magB2);
	}

	/**
	 * TODO:
	 * Computes a similarity score between two bags-of-words. Use one of the above options 
	 * (cosine or Jaccard).
	 * 
	 * compute similarity between two bags-of-words.
	 * @param bag1 first bag of words
	 * @param bag2 second bag of words
	 * @param sim_opt COSINE or JACCARD similarity
	 * @return similarity score
	 */
	private double similarity(ArrayList<String> bag1, ArrayList<String> bag2, String sim_opt) {
		if(sim_opt.equals("JACCARD")){
			return jaccardSimilarity(bag1, bag2);
		}else if(sim_opt.equals("COSINE")){
			return cosineSimilarity(bag1, bag2);
		}else if(sim_opt.equals("NAIVE")){
			//Will default to most frequent sense. This is the base line
			//Average		0.6123162098808664	0.30306689286977656	0.3693343514114866
			return 0;
		}
		System.out.println("ERROR! -- sim_opt not recognized");
		return 0;
	}
	
	/**
	 * TODO:
	 * For each target word (from ambiguousWords) in each sentence (from testCorpus), create
	 * a context and all possible senses (using getSignatures) for the target. For each sense, use
	 * similarity to find the similarity score between the signature and context. Map each sense
	 * key to its similarity score in a HashMap object, which is then inserted into an ArrayList,
	 * which is then inserted into this.predictions.
	 * You need to allow different options for context construction and similarity metric though
	 * the arguments. Jaccard similarity using ALL_WORDS and ALL_WORDS_R is required.
	 * 
	 * This is the WSD function that prediction what senses are more likely.
	 * @param context_option: one of {ALL_WORDS, ALL_WORDS_R, WINDOW, POS}
	 * @param window_size: an odd positive integer > 1
	 * @param sim_option: one of {COSINE, JACCARD}
	 */
	public void predict(String context_option, int window_size, String sim_option) {
		//getSignatures("own", "ADJ");
		// ISenseKey sk = SenseKeyParser.getInstance().parseLine("once%4:02:02::");

		// IWord word = wordnetdict.getWord(sk);
		// System.out.println(word.getPOS());
		
		for(int i = 0; i < testCorpus.size(); i++){
			Sentence sentence = testCorpus.get(i);
			ArrayList<Pair<String, String>> aWords = ambiguousWords.get(i);
			ArrayList<Integer> locations =  ambiguousLocations.get(i);
			//ArrayList<HashMap<String, Double>> sentencePredictions = new ArrayList<HashMap<String, Double>>(testCorpus.size());
			ArrayList<ArrayList<SenseScore>> sentencePredictions = new ArrayList<ArrayList<SenseScore>>(testCorpus.size());
			for(int j = 0; j < aWords.size(); j++){
				Pair<String, String> wordPOS = aWords.get(j);
				if(wordPOS.getKey() == null || wordPOS.getValue() == null){
					System.out.println(sentence.getWordAt(ambiguousLocations.get(i).get(j)));
					continue;
				}
				Map<String, Pair<String, Integer> > senses = getSignatures(wordPOS.getKey(), wordPOS.getValue());
				int targetIndex = locations.get(j);
				ArrayList<String> context = getContext(sentence, context_option, window_size, targetIndex);
				String[] senseIterator = senses.keySet().toArray(new String[0]);
				double maxSim = 0;
				int maxIndex = 0;
				//HashMap<String, Double> senseMap = new HashMap<String, Double>();
				ArrayList<SenseScore> wordPredictions = new ArrayList<SenseScore>();
				for(int k = 0; k < senseIterator.length; k++){
					Pair<String, Integer> sense = senses.get(senseIterator[k]);
					ArrayList<String> signature = str2bow(sense.getKey(), context_option);
					double sim = similarity(context, signature, sim_option);
					SenseScore ss = new SenseScore(senseIterator[k], sense.getValue(), sim);
					//wordPredictions.add(ss);
					boolean added = false;
					for(int h = 0; h < wordPredictions.size(); h++){
						if(SenseScore.compare(ss, wordPredictions.get(h)) > 0 ){
							//System.out.println("int " + ss.toString() +" > " + wordPredictions.get(h).toString());
							wordPredictions.add(h, ss);
							added = true;
							
							break;
						}
					}
					if(!added){
						wordPredictions.add(ss);
						//System.out.println("end " + ss.toString() +" < everything");
						//System.out.println("Adding on fre " + ss.senseKey);
					} 
					// senseMap.put(senseIterator[k], sim);
					// if(maxSim < sim){
					// 	maxSim = sim;
					// 	maxIndex = k;
					// }
				}

				//Record predictions
				sentencePredictions.add(wordPredictions);
			}
			this.predictions.add(sentencePredictions);
		}
	}
	

	/**
	 * Multiple senses are concatenated using comma ",". Separate them out.
	 * @param senses
	 * @return
	 */
	private ArrayList<String> parseSenseKeys(String senseStr) {
		ArrayList<String> senses = new ArrayList<String>();
		String[] items = senseStr.split(",");
		for (String item : items) {
			senses.add(item);
		}
		return senses;
	}
	
	/**
	 * TODO:
	 * Use this.predictions and this.groundTruths to generate precision, recall and F1 score
	 * at the top K positions. Note that these 3 metrics are calculated for a target word.
	 * precision=(# of correctly predicted senses with the largest K similarities) / K
	 * recall=(# of correctly predicted senses with the largest K similarities) / (# sense keys of the target)
	 * f1=(2 x precision x recall) / (precision + recall)
	 * A sense key predicted for a target word is considered correct if it matches any one of the
	 * sense keys in the ground truth of the target word (note that each target word can have
	 * correct multiple WordNet senses).
	 * 
	 * Precision/Recall/F1-score at top K positions
	 * @param groundTruths: a list of sense id strings, such as [become%2:30:00::, become%2:42:01::]
	 * @param predictions: a map from sense id strings to the predicted similarity
	 * @param K
	 * @return a list of [top K precision, top K recall, top K F1]
	 */
	private ArrayList<Double> evaluateWord(ArrayList<String> groundTruths, ArrayList<SenseScore> predictions, int k) {
		if(predictions == null || groundTruths == null){
			//Error
			System.out.println("EXTRA BAD");
			return null;
		}else if(groundTruths.size() == 0){
			//Error
			System.out.println("BAD TRUTH");
			return null;
		}else if(predictions.size()==0){
			//Error
			//System.out.println("BAD PREDICTION");
			return null;
		}
		//System.out.println("---------------------------GOOD----------------------------------");
		
		int correctPredictions = 0;
		int max = k < predictions.size() ? k : predictions.size();
		for(int i = 0; i < max; i++){
			if(groundTruths.contains(predictions.get(i).senseKey))
				correctPredictions++;
		}
		
		double precision = correctPredictions / (double)k;
		double recall = correctPredictions / (double)groundTruths.size();

		double f1 = 0.0;
		if(recall != 0.0 && precision != 0.0){
			f1 = (2.0 * precision * recall) / (precision + recall);
			//System.out.println("F1 = " + f1 + " : " + (2.0 * precision * recall) + " / " + (precision + recall));
		} 

		ArrayList<Double> results = new ArrayList<Double>(3);
		results.add(precision);
		results.add(recall);
		results.add(f1);

		return results;

	}

	private ArrayList<Double> evaluateSentence(ArrayList<String> groundTruths, ArrayList<ArrayList<SenseScore>> predictions, int k) {
		int size = 0;
		if(predictions.size() == groundTruths.size()){
			size = predictions.size();
		}else{
			return null;
		}
		
		int normalizer = size;

		double[] results = new double[3];
		for(int i = 0; i < size; i++){
			ArrayList<SenseScore> wordPrediction = predictions.get(i);
			ArrayList<String> gTruth = parseSenseKeys(groundTruths.get(i));
			ArrayList<Double> wordResults = evaluateWord(gTruth, wordPrediction, k);

			if(wordResults != null && wordResults.size() == 3){
				results[0] += wordResults.get(0);
				results[1] += wordResults.get(1);
				results[2] += wordResults.get(2);
			}else{
				//error
				normalizer--;
			}
		}

		if(normalizer <= 0){
			return null;
		}

		ArrayList<Double> out = new ArrayList<Double>(3);
		out.add(results[0] / (double)normalizer);
		out.add(results[1] / (double)normalizer);
		out.add(results[2] / (double)normalizer);
		return out;

	}
	
	/**
	 * TODO:
	 * Call the above function to calculate and then average the 3 metrics over all targets.
	 * 
	 * Test the prediction performance on all test sentences
	 * @param k Top-K precision/recall/f1
	 */
	public ArrayList<Double> evaluate(int k) {
		int size = 0;
		if(this.predictions.size() == this.groundTruths.size()){
			size = this.predictions.size();
		}else{
			return null;
		}

		int normalizer = size;
		double[] results = new double[3];
		for(int i = 0; i < size; i++){
			//ArrayList<HashMap<String, Double>> sentencePrediction = this.predictions.get(i);
			ArrayList<ArrayList<SenseScore>> sentencePrediction = this.predictions.get(i);
			ArrayList<String> gTruth = this.groundTruths.get(i);
			ArrayList<Double> sentenceResults = evaluateSentence(gTruth, sentencePrediction, k);

			if(sentenceResults != null && sentenceResults.size() == 3){
				results[0] += sentenceResults.get(0);
				results[1] += sentenceResults.get(1);
				results[2] += sentenceResults.get(2);
			}else{
				//error
				normalizer--;
			}
			//check overflow
			
		}
		ArrayList<Double> out = new ArrayList<Double>(3);
		out.add(results[0] / (double)normalizer);
		out.add(results[1] / (double)normalizer);
		out.add(results[2] / (double)normalizer);
		return out;
	}

	private static boolean setup(){
		wordnetdict = initDictionary();
		if(wordnetdict == null){
			System.out.println("ERROR reading dictionary");
			return false;
		}

		stopwords = readStopWords();
		if(stopwords == null){
			System.out.println("ERROR reading stopwords");
			return false;
		}

		pipeline = initPipeline();
		if(pipeline == null){
			System.out.println("ERROR setting up NLP pipeline");
			return false;
		}

		return true;
	}

	private static void printResults(ArrayList<Double> results, String filename){
		System.out.print(filename);
		System.out.print("\t");
		if(results != null || results.size() < 3){
			System.out.print("\t");
			System.out.print(results.get(0));
			System.out.print("\t");
			System.out.print(results.get(1));
			System.out.print("\t");
			System.out.println(results.get(2));
			
			
		}else{
			System.out.print("Incomplete algorithm.");
			System.out.print("\t\n");
		}
	}

	private static ArrayList<Double> processFile(String filename){
		Lesk model = new Lesk();
		try {
			model.readTestData(filename);
		} catch (Exception e) {
			System.out.println(filename);
			e.printStackTrace();
			return null;
		}
		String context_opt = Lesk.ALL_WORDS;
		int window_size = 3;
		String sim_opt = "COSINE";
		
		model.predict(context_opt, window_size, sim_opt);
		
		ArrayList<Double> res = model.evaluate(1);
		printResults(res, filename);
		return res;
	}

	private static void progress(int cur, int max){
		progress((double)cur, (double)max, cur == max);
	}

	private static void progress(double cur, double max, boolean done){
		int length = 50;
		int inc = (int)Math.round((cur/max) * length);
		StringBuilder output = new StringBuilder();
		output.append("|");
		for(int i = 0; i < length; i++){
			if(i < inc) output.append("#");
			else output.append(" ");
		}
		if(done){
			output.append(String.format("| done \n", (100.0 * cur/max)));
		}else{
			output.append(String.format("| %.1f%% \r", (100.0 * cur/max)));
		}
		System.err.print(output.toString());
	}

	/**
	 * @param args[0] file name of a test corpus
	 */
	public static void main(String[] args) {
		if(setup()){
			int normalizer = args.length;
			double[] results = new double[3];
			int max = args.length;
			int cur = 0;
			for(String file : args){
				ArrayList<Double> res = processFile(file);
				if(res != null && res.size() == 3){
					results[0] += res.get(0);
					results[1] += res.get(1);
					results[2] += res.get(2);
				}else{
					normalizer--;
				}
				cur++;
				progress(cur, max);
			}
			ArrayList<Double> avg = new ArrayList<Double>(3);
			avg.add(results[0] / (double)normalizer);
			avg.add(results[1] / (double)normalizer);
			avg.add(results[2] / (double)normalizer);

			printResults(avg, "Average");
		}
	}
}
