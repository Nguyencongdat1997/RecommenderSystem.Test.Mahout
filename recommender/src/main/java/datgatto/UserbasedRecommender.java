package datgatto;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericBooleanPrefDataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UserbasedRecommender {

    public static void main(String[] args) throws IOException, TasteException {

        Logger log = LoggerFactory.getLogger(UserbasedRecommender.class);

        DataModel model = new FileDataModel(new File("input/foody_rate.csv"));
        
        //PearsonCorrelationSimilarity
        RecommenderBuilder builder = new RecommenderBuilder() {
            public Recommender buildRecommender(DataModel dataModel) throws TasteException{
                UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
                UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, dataModel);
                return new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
            }
        };

        RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();        
        double result = evaluator.evaluate(builder, null, model, 0.7, 0.2);
        System.out.print("PearsonCorrelationSimilarity" );
        System.out.println(result);

        //EuclideanDistanceSimilarity
        builder = new RecommenderBuilder() {
            public Recommender buildRecommender(DataModel dataModel) throws TasteException{
                UserSimilarity similarity = new EuclideanDistanceSimilarity(dataModel);
                UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, dataModel);
                return new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
            }
        };

        evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();        
        result = evaluator.evaluate(builder, null, model, 0.7, 0.2);
        System.out.print("EuclideanDistanceSimilarity" );
        System.out.println(result);

    }
}