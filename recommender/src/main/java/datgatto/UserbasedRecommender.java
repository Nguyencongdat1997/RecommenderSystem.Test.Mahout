package datgatto;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UserbasedRecommender {

    public static void main(String[] args) throws IOException, TasteException {

        Logger log = LoggerFactory.getLogger(UserbasedRecommender.class);

        // Load data
        DataModel model = new FileDataModel(new File("input/u.user"));

        //Model
        RecommenderBuilder builder = new RecommenderBuilder() {
            public Recommender buildRecommender(DataModel model) {
                // Compute the similarity between users, according to their preferences
                UserSimilarity userSimilarity = new PearsonCorrelationSimilarity(model);

                // Neigborhood
                UserNeighborhood neighborhood = new NearestNUserNeighborhood(3, userSimilarity, model);

                // Recommender
                Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, userSimilarity);
                
                return recommender;
            }
        };

        //Evaluate
        RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
        double evaluation = evaluator.evaluate(builder, model, 0.9, 1.0);
        System.out.println(Double.toString(evaluation));
    }
}