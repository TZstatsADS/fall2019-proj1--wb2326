# What Song Lyrics Tell Us About?

### Summary

In this project, I explore the texts from lyrics, a corpus of over 380,000 lyrics form MetrolLyrics. I want to find out what genre and what artist are most productive. Does their productivity decay or increase as time goes by? Do Rockstars complain more or instruct more? How about Rapstars? If complaining, what do they hate?

### Data preparation and basic analysis

There are 125715 observations and 7 features in this dataset. There are 12 genres of music in this dataset such as Hip-Hop, Other, Pop, Metal, Rock... There are 2534 artists recorded in this dataset such as a, a-boogie-wit-da-hoodie, a-camp, a-canorous-quintet, a-change-of-pace...

+ Check the word counts by genre & word counts histogram.
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/word_count_by_genre.png)
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/output_18_1.png)

The histogram clustered around 200 meaning most songs have about 200 words, while the distribution skewed to the right, we know there’re some extreme outliers with a large number of words. 

Before digging into the details of genre/artist differences, let's first take a look at the song productivity of decades from 1960s to 2010s. Here we can see that the song productivity raised a lot since 1960s and reached summit at 2000s. However, the data were collected in mid 2010s thus many songs produced after that might not have been recorded.  
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/output_19_0.png)

Overall, Hip-hop uses the greatest average amount of words, which makes sense as rap “utter[s] sharply or vigorously: [rap] out a command”. In contrast, Jazz employees the least words.

### Top 5 Most Productive musicians

Here are the top 5 most productive musicians along with their genres.

![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/top5.png)

![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/output_22_1.png)

To get more details about the lyrics text of the 5 most eloquent artist, I created 5 corresponding wordclouds, which tell us the most frequent words appeared in artists’ lyrics: “love”, “time”, “baby”, “well’…, are the popular words poping up frequently in lyrics.

dolly-parton
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/output_30_0.png)

elton-john
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/output_31_0.png)

chris-brown
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/output_32_0.png)

bee-gees
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/output_33_0.png)

bob-dylan
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/output_34_0.png)

However, there is one great distinction between rapstars and other genre artists. They mentioned “girl” almost as twice as frequent.

Top 10 most frequent words for each genre:

Country
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/Country.png)

Electronic
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/Electronic.png)

Folk
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/Folk.png)

Hip-Hop
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/Hip-Hop.png)

Indie
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/Indie.png)

Jazz
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/Jazz.png)

Metal
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/Metal.png)

Not Available
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/Not%20Available.png)

Other
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/Other.png)

Pop
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/Pop.png)

R&B!
[](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/R%26B.png)

Rock
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/Rock.png)

![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/words_ratio.png)

### Does productivity change along time?
+ time series plot

Finally, I wanted to explore the relationship between time and productivity. I tracked the top 5 most productive musicians’ (mentioned above) song counts by decade. 

dolly-parton
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/output_35_1.png)

elton-john
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/output_36_1.png)

chris-brown
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/output_37_1.png)

bee-gees
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/output_38_1.png)

bob-dylan
![](https://github.com/TZstatsADS/fall2019-proj1--wb2326/blob/master/figs/output_39_1.png)

Except rapstar Chris Brown started producing a bit late since 2000s, all other artists have long and stable (regular ups and downs) creation rates.  
