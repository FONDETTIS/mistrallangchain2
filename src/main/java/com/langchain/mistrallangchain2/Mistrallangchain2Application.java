package com.langchain.mistrallangchain2;

import org.springframework.ai.transformer.splitter.TextSplitter;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

//import dev.langchain4j.model.ChatLanguageModel;
//import dev.langchain4j.model.openai.OpenAiChatModel;
//import dev.langchain4j.retriever.EmbeddingStoreInMemoryRetriever;
//import dev.langchain4j.retriever.EmbeddingStoreRetriever;
//import dev.langchain4j.retriever.documentcontent.DocumentContentRetriever;
//import dev.langchain4j.store.EmbeddingStore;
//import dev.langchain4j.store.inmemory.InMemoryEmbeddingStore;
//import dev.langchain4j.embedding.EmbeddingModel;
//import dev.langchain4j.embedding.openai.OpenAiEmbeddingModel;

import dev.langchain4j.model.mistralai.MistralAiChatModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.mistralai.MistralAiChatModel.MistralAiChatModelBuilder;
import dev.langchain4j.model.mistralai.MistralAiChatModelName;

import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.agent.tool.ReturnBehavior;                                             
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.segment.TextSegment;

import org.springframework.ai.transformer.splitter.TextSplitter;

import java.util.List;
import java.util.Map;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.mistralai.MistralAiChatModel;
import dev.langchain4j.model.mistralai.MistralAiEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static dev.langchain4j.model.mistralai.MistralAiChatModelName.MISTRAL_MEDIUM_LATEST;
import static dev.langchain4j.model.mistralai.MistralAiEmbeddingModelName.MISTRAL_EMBED;
import static java.util.stream.Collectors.joining;

@SpringBootApplication(exclude = {
		org.springframework.ai.model.chat.client.autoconfigure.ChatClientAutoConfiguration.class,
		org.springframework.ai.model.mistralai.autoconfigure.MistralAiChatAutoConfiguration.class,
		org.springframework.ai.model.mistralai.autoconfigure.MistralAiEmbeddingAutoConfiguration.class,
		org.springframework.ai.model.mistralai.autoconfigure.MistralAiModerationAutoConfiguration.class,
		org.springframework.ai.model.mistralai.autoconfigure.MistralAiOcrAutoConfiguration.class
	    //OpenAiAutoConfiguration.class,
	    //AzureOpenAiAutoConfiguration.class
	} , scanBasePackages = "com.langchain.mistrallangchain2")
public class Mistrallangchain2Application {

	

	
	public static void main(String[] args) {
		SpringApplication.run(Mistrallangchain2Application.class, args);
		
		System.out.println("MISTRALLANGCHAIN2-MAIN1");
		
		
		//Ces quelques lignes sont tirées d'un autre exemple.
		// Load a text file (or URL)
		//List<Document> documents = new TextDocumentLoader("your_file.txt").load();
		//List<Document> webPages = new WebPageDocumentLoader("https://example.com").load();
		// Split documents into chunks (for better retrieval)
		//TextSplitter splitter = new TextSplitter();
		//List<Document> chunks = splitter.splitDocuments(documents);
		
		
		
		// In this very simple example, we are getting data that we want to use for RAG.
        // We will use a history about origin of the Llama by National Geographic https://www.nationalgeographic.es/animales/llama.
        //Document document = loadDocument(toPath("Documents/story-about-origin-of-the-llama.txt"), new TextDocumentParser());
        Document document = FileSystemDocumentLoader.loadDocument("/home/broaduser/Documents/story-about-origin-of-the-llama.txt");

        // In a RAG system, it is crucial to split the document into smaller chunks so that it's more effective
        // to identify and retrieve the most relevant information in the retrieval process later
        DocumentSplitter splitter = DocumentSplitters.recursive(200, 0);
        List<TextSegment> segments = splitter.split(document);
        
        
        // Now, for each text segment, we need to create text embeddings, which are numeric representations of the text in the vector space.
        // Of course, we will use Mistral AI for this purpose.
        EmbeddingModel embeddingModel = MistralAiEmbeddingModel.builder()
                .apiKey(System.getenv("MISTRAL_AI_API_KEY"))
                .modelName(MISTRAL_EMBED)
                .build();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        
        System.out.println("MISTRALLANGCHAIN2-MAIN2");

        // Once we get the text embeddings, we will store them in a vector database for efficient processing and retrieval.
        // For simplicity, this example uses an in-memory store, but you can choose any external compatible store for production environments.
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        
        System.out.println("MISTRALLANGCHAIN2-MAIN3");

        // Whenever users ask a question, we also need to create embeddings for this question using the same embedding models as before.
        String question = "Who create the llamas?";
        Embedding questionEmbedding = embeddingModel.embed(question).content();

        System.out.println("MISTRALLANGCHAIN2-MAIN4");
        
        // We can perform a search on the vector database and retrieve the most relevant text chunks based on the user question.
        EmbeddingSearchRequest embeddingSearchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(questionEmbedding)
                .maxResults(3)
                .minScore(0.7)
                .build();
        List<EmbeddingMatch<TextSegment>> relevantEmbeddings = embeddingStore.search(embeddingSearchRequest).matches();

        System.out.println("MISTRALLANGCHAIN2-MAIN5");
        
        // Now we can offer the relevant information as the context information within the prompt.
        // Here is a prompt template where we can include both the retrieved text and user question in the prompt.
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Context information is below.:\n"
                        + "------------------\n"
                        + "{{information}}\n"
                        + "------------------\n"
                        + "Given the context information and not prior knowledge, answer the query.\n"
                        + "Query: {{question}}\n"
                        + "Answer:");
        String information = relevantEmbeddings.stream()
                .map(match -> match.embedded().text())
                .collect(joining("\n\n"));
        
        System.out.println("MISTRALLANGCHAIN2-MAIN6");

        Map<String, Object> promptInputs = new HashMap<>();
        promptInputs.put("question", question);
        promptInputs.put("information", information);

        Prompt prompt = promptTemplate.apply(promptInputs);

        // Now we can use the Mistral AI chat model to generate the answer to the user question based on the context information.
        ChatModel chatModel = MistralAiChatModel.builder()
                .apiKey(System.getenv("MISTRAL_AI_API_KEY"))
                .modelName(MISTRAL_MEDIUM_LATEST)
                .temperature(0.2) // expect a more focused and deterministic answer
                .logRequests(true)
                .logResponses(true)
                .build();
        
        System.out.println("MISTRALLANGCHAIN2-MAIN7");

        //AiMessage aiMessage = chatModel.chat(prompt.toUserMessage()).aiMessage();
        
        System.out.println("MISTRALLANGCHAIN2-MAIN8");
        
        //String answer = aiMessage.text();
        //System.out.println(answer); // According to Inca legend, the llamas were created by the mythical founders of the Inca Empire....
    }
	
	
	//Cette méthode a été contournée, elle ne sert donc à rien.
	static Path toPath(String fileName) {
        try {
            URL fileUrl = Mistrallangchain2Application.class.getResource(fileName);
            return Paths.get(fileUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

	
}

		


