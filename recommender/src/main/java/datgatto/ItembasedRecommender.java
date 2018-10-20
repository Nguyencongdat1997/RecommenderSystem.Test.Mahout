package datgatto;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ItembasedRecommender {

    public static void main(String[] args) throws IOException, TasteException {

        Logger log = LoggerFactory.getLogger(ItembasedRecommender.class);

        // Load historical data about user preferences
        DataModel model = new FileDataModel(new File("input/u.user"));
        ItemSimilarity itemSimilarity = new EuclideanDistanceSimilarity (model);
        Recommender itemRecommender = new GenericItemBasedRecommender(model,itemSimilarity);
        List<RecommendedItem> itemRecommendations = itemRecommender.recommend(3, 2);
        for (RecommendedItem itemRecommendation : itemRecommendations) {
        System.out.println("Item: " + itemRecommendation);
    }
}