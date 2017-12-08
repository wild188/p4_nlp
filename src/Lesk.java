/**
 * Implement the Lesk algorithm for Word Sense Disambiguation (WSD)
 */
import java.util.*;
import java.io.*;
import javafx.util.Pair;

import edu.mit.jwi.*;
import edu.mit.jwi.Dictionary;
import edu.mit.jwi.item.*; 

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
	private ArrayList<ArrayList<HashMap<String, Double> > > predictions = new ArrayList<ArrayList<HashMap<String, Double> > >();
	
	/**
	 * Each entry is a list of ground truth senses for the ambiguous locations.
	 * Each String object can contain multiple synset ids, separated by comma.
	 * E.g.,
	 * 		groundTruths.get(0) is a list of strings ["become%2:30:00::,become%2:42:01::"]
	 * 		groundTruths.get(1) is a list of strings ["take%2:30:01::,take%2:38:09::,take%2:38:10::,take%2:38:11::,take%2:42:10::", "apply%2:40:00::"]
	 */
	private ArrayList<ArrayList<String> > groundTruths = new ArrayList<ArrayList<String> >();
	
	/* This section contains the NLP tools */
	
	private static final Set<String> POS = new HashSet<String>(Arrays.asList("ADJECTIVE", "ADVERB", "NOUN", "VERB"));
	
	private static IDictionary wordnetdict;
	
	private StanfordCoreNLP pipeline;

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

	/**
	 * TODO:
	 * The constructor initializes any WordNet/NLP tools and reads the stopwords.
	 */
	public Lesk() {
		//Dictionary and stopwords already done statically

		// Create StanfordCoreNLP object properties, with POS tagging
        // (required for lemmatization), and lemmatization
        Properties props;
        props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma");

        // StanfordCoreNLP loads a lot of models, so you probably
        // only want to do this once per execution
        this.pipeline = new StanfordCoreNLP(props);
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
		} else if (pos.equals("NOUN") || pos.equals("VERB")) {
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
		System.out.println(mySentence.toString());
	}

	private int sentenceCount;

	private void processLine(String line){
		if(line.charAt(0) == '#'){
			//The line is a sense definition

		}else{
			String[] words = line.split(" ");
			int index = -1;
			try{
				index = Integer.parseInt(words[0]);
			}catch(NumberFormatException nfe){
				index = -1;
			}
			if(index < 0){
				//The line is a new sentence
				inputSentence(line);
				sentenceCount++;
			}else{
				//The line is the index of the ambiguous word

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
		sentenceCount = 0;

		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line = br.readLine();
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
		return null;
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
	private ArrayList<String> str2bow(String str) {
		return null;
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
	private ArrayList<Double> evaluate(ArrayList<String> groundTruths, HashMap<String, Double> predictions, int K) {
		return null;
	}
	
	/**
	 * TODO:
	 * Call the above function to calculate and then average the 3 metrics over all targets.
	 * 
	 * Test the prediction performance on all test sentences
	 * @param K Top-K precision/recall/f1
	 */
	public ArrayList<Double> evaluate(int K) {
		return null;
	}

	private static void setup(){
		wordnetdict = initDictionary();
		if(wordnetdict == null){
			System.out.println("ERROR reading dictionary");
			return;
		}

		stopwords = readStopWords();
		if(stopwords == null){
			System.out.println("ERROR reading stopwords");
			return;
		}
	}

	private static void processFile(String filename){
		Lesk model = new Lesk();
		try {
			model.readTestData(filename);
		} catch (Exception e) {
			System.out.println(filename);
			e.printStackTrace();
		}
		String context_opt = "ALL_WORDS";
		int window_size = 3;
		String sim_opt = "JACCARD";
		
		model.predict(context_opt, window_size, sim_opt);
		
		ArrayList<Double> res = model.evaluate(1);
		System.out.print(filename);
		System.out.print("\t");
		if(res != null){
			System.out.print("\t");
			System.out.print(res.get(0));
			System.out.print("\t");
			System.out.print(res.get(1));
			System.out.print("\t");
			System.out.println(res.get(2));
		}else{
			System.out.print("Incomplete algorithm.");
			System.out.print("\t\n");
		}
	}

	/**
	 * @param args[0] file name of a test corpus
	 */
	public static void main(String[] args) {
		setup();
		for(String file : args){
			processFile(file);
		}
	}
}
