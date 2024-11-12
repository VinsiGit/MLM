# DistilBERT Masked Language Model (MLM) Project

## Inhoudsopgave
1. [Beschrijving van de gekozen SSL-methode en pretext task](#beschrijving-van-de-gekozen-ssl-methode-en-pretext-task)
2. [Dataset Overzicht en Voorbereiding voor SSL](#dataset-overzicht-en-voorbereiding-voor-ssl)
3. [Gebruik](#gebruik)
4. [Conclusie](#conclusie)

## Beschrijving van de gekozen SSL-methode en pretext task

In dit project maken wij gebruik van **Self-Supervised Learning (SSL)** met het **DistilBERT**-model voor **Masked Language Modeling (MLM)**. De gekozen pretext task is het voorspellen van gemaskeerde tokens binnen een tekst. Dit proces omvat het volgende stappenplan:

- **Maskeren van tokens**: Bepaalde woorden in een zin worden vervangen door het `[MASK]`-token.
- **Voorspellen van gemaskeerde tokens**: Het model wordt getraind om de oorspronkelijke woorden nauwkeurig te voorspellen op basis van de context waarin ze voorkomen.

Deze methode stelt het model in staat om diepgaande taalrepresentaties te leren zonder de noodzaak van gelabelde data.

## Dataset Overzicht en Voorbereiding voor SSL

Voor dit project hebben we de **"wikitext-2-raw-v1"** dataset gebruikt, beschikbaar via het [`datasets`](https://huggingface.co/docs/datasets/) pakket. Deze dataset bevat onbewerkte tekst uit Wikipedia-artikelen, wat het ideaal maakt voor taalmodelleringstaken zoals MLM.

### Voorbereiding van de Dataset

De dataset is op de volgende manier voorbereid voor SSL:

1. **Tokenisatie**:
    - De tekstdata is getokeniseerd met behulp van de **DistilBERT-tokenizer**.
    - Zinnen worden opgesplitst in tokens en geconverteerd naar een formaat dat door het model kan worden verwerkt.
    
2. **Padding en Truncatie**:
    - Om uniforme invoerlengtes te waarborgen, worden zinnen gepad of afgekapt tot een maximale lengte van **128 tokens**.
    
3. **Data Collatie**:
    - Een **data collator** is gebruikt om batches van getokeniseerde zinnen samen te stellen.
    - Willekeurig tokens worden gemaskeerd met een kans van **15%** om de MLM-taak toe te passen.

Deze stappen zorgen ervoor dat de dataset geschikt is voor het trainen van het model op de MLM-taak.

## Uitleg van Resultaten en Verbeterpunten

Na het trainen van het model hebben we de prestaties geëvalueerd op de validatieset. De resultaten tonen aan dat het model effectief gemaskeerde woorden kan voorspellen, wat aangeeft dat het taalbegrip is verbeterd. Hier volgt een nadere toelichting op de behaalde resultaten en mogelijke verbeterpunten:

### Resultaten

- **Nauwkeurigheid**: Het model presteert met een redelijke nauwkeurigheid bij het voorspellen van gemaskeerde tokens.
- **Perplexity**: De perplexiteitsmeting bevestigt dat het model lagere verwarringsscores heeft, wat wijst op betere voorspellingen.
- **Visualisatie**: De training loss over de epochs toont een consistente afname, wat duidt op effectieve training.


### Conclusie
Dit project demonstreert de toepassing van Self-Supervised Learning met het DistilBERT-model voor Masked Language Modeling. Door gebruik te maken van de "wikitext-2-raw-v1" dataset en MLflow voor experiment tracking, hebben we een effectief taalmodel ontwikkeld dat gemaskeerde tokens nauwkeurig kan voorspellen.

### Visuele Weergave van Modelprestaties

![Training Loss](images/training_loss.png)
![Validatie Perplexity](images/validation_perplexity.png)