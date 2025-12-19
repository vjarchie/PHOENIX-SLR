# -*- coding: utf-8 -*-
"""
German Sign Language Gloss to English Translation.

Provides translation from DGS glosses to English for the PHOENIX weather domain.
"""

# Common PHOENIX weather glosses to English
GLOSS_TO_ENGLISH = {
    # Weather conditions
    'SONNE': 'sun',
    'SONNIG': 'sunny',
    'REGEN': 'rain',
    'REGNEN': 'raining',
    'WOLKE': 'cloud',
    'WOLKEN': 'clouds',
    'WOLKIG': 'cloudy',
    'NEBEL': 'fog',
    'SCHNEE': 'snow',
    'SCHNEIEN': 'snowing',
    'GEWITTER': 'thunderstorm',
    'STURM': 'storm',
    'WIND': 'wind',
    'WINDIG': 'windy',
    'WARM': 'warm',
    'KALT': 'cold',
    'KUEHL': 'cool',
    'HEISS': 'hot',
    'FROST': 'frost',
    'EIS': 'ice',
    'HAGEL': 'hail',
    'TROCKEN': 'dry',
    'FEUCHT': 'humid',
    'NASS': 'wet',
    'WECHSELHAFT': 'changeable',
    'BEWOELKT': 'overcast',
    'FREUNDLICH': 'pleasant',
    'SCHAUER': 'showers',
    
    # Time
    'MORGEN': 'tomorrow',
    'HEUTE': 'today',
    'GESTERN': 'yesterday',
    'UEBERMORGEN': 'day-after-tomorrow',
    'JETZT': 'now',
    'SPAETER': 'later',
    'FRUEH': 'early',
    'ABEND': 'evening',
    'NACHT': 'night',
    'MITTAG': 'noon',
    'NACHMITTAG': 'afternoon',
    'VORMITTAG': 'morning',
    'WOCHENENDE': 'weekend',
    
    # Days
    'MONTAG': 'Monday',
    'DIENSTAG': 'Tuesday',
    'MITTWOCH': 'Wednesday',
    'DONNERSTAG': 'Thursday',
    'FREITAG': 'Friday',
    'SAMSTAG': 'Saturday',
    'SONNTAG': 'Sunday',
    
    # Months
    'JANUAR': 'January',
    'FEBRUAR': 'February',
    'MAERZ': 'March',
    'APRIL': 'April',
    'MAI': 'May',
    'JUNI': 'June',
    'JULI': 'July',
    'AUGUST': 'August',
    'SEPTEMBER': 'September',
    'OKTOBER': 'October',
    'NOVEMBER': 'November',
    'DEZEMBER': 'December',
    
    # Directions/Regions
    'NORD': 'north',
    'SUED': 'south',
    'OST': 'east',
    'WEST': 'west',
    'NORDOST': 'northeast',
    'NORDWEST': 'northwest',
    'SUEDOST': 'southeast',
    'SUEDWEST': 'southwest',
    'MITTE': 'center',
    'REGION': 'region',
    'BEREICH': 'area',
    'GEBIET': 'area',
    'KUESTE': 'coast',
    'ALPEN': 'Alps',
    'BERG': 'mountain',
    'FLUSS': 'river',
    'MEER': 'sea',
    'DEUTSCHLAND': 'Germany',
    
    # Numbers
    'NULL': 'zero',
    'EINS': 'one',
    'ZWEI': 'two',
    'DREI': 'three',
    'VIER': 'four',
    'FUENF': 'five',
    'SECHS': 'six',
    'SIEBEN': 'seven',
    'ACHT': 'eight',
    'NEUN': 'nine',
    'ZEHN': 'ten',
    'ELF': 'eleven',
    'ZWOELF': 'twelve',
    'DREIZEHN': 'thirteen',
    'VIERZEHN': 'fourteen',
    'FUENFZEHN': 'fifteen',
    'SECHZEHN': 'sixteen',
    'SIEBZEHN': 'seventeen',
    'ACHTZEHN': 'eighteen',
    'NEUNZEHN': 'nineteen',
    'ZWANZIG': 'twenty',
    'DREISSIG': 'thirty',
    
    # Temperature
    'GRAD': 'degrees',
    'TEMPERATUR': 'temperature',
    'MAXIMAL': 'maximum',
    'MINIMAL': 'minimum',
    'MINUS': 'minus',
    'PLUS': 'plus',
    
    # Actions/Verbs
    'KOMMEN': 'come',
    'GEHEN': 'go',
    'BLEIBEN': 'stay',
    'WERDEN': 'become',
    'SEIN': 'be',
    'HABEN': 'have',
    'ZEIGEN': 'show',
    'SEHEN': 'see',
    'SCHEINEN': 'shine',
    'FALLEN': 'fall',
    'STEIGEN': 'rise',
    'SINKEN': 'drop',
    'WEHEN': 'blow',
    'REGNEN': 'rain',
    'SCHNEIEN': 'snow',
    
    # Adjectives/Adverbs
    'MEHR': 'more',
    'WENIG': 'little',
    'WENIGER': 'less',
    'VIEL': 'much',
    'BISSCHEN': 'bit',
    'STARK': 'strong',
    'SCHWACH': 'weak',
    'LEICHT': 'light',
    'HOCH': 'high',
    'TIEF': 'low',
    'SELTEN': 'rare',
    'OFT': 'often',
    'MANCHMAL': 'sometimes',
    'IMMER': 'always',
    'NIE': 'never',
    'MOEGLICH': 'possible',
    'WAHRSCHEINLICH': 'probably',
    'BESONDERS': 'especially',
    'UEBERALL': 'everywhere',
    'TEIL': 'part',
    'GANZ': 'whole',
    
    # Connectors
    'UND': 'and',
    'ODER': 'or',
    'ABER': 'but',
    'AUCH': 'also',
    'NUR': 'only',
    'NOCH': 'still',
    'SCHON': 'already',
    'DANN': 'then',
    'DANACH': 'after',
    'BIS': 'until',
    'VON': 'from',
    'NACH': 'to',
    'IN': 'in',
    'AUF': 'on',
    'MIT': 'with',
    'DURCH': 'through',
    'DURCHGEHEND': 'continuously',
    
    # Questions
    'WIE': 'how',
    'WAS': 'what',
    'WO': 'where',
    'WANN': 'when',
    'WARUM': 'why',
    
    # Other common
    'WETTER': 'weather',
    'AUSSEHEN': 'look',
    'WIE-AUSSEHEN': 'look-like',
    'VORHERSAGE': 'forecast',
    'PROGNOSE': 'prediction',
    'UEBERSCHWEMMUNG': 'flood',
    'HOCHWASSER': 'high-water',
    'TAGESSCHAU': 'news',
    'LIEB': 'dear',
    'ZUSCHAUER': 'viewers',
    'FREUEN': 'happy',
    'KOENNEN': 'can',
    'SOLLEN': 'should',
    'MUESSEN': 'must',
    'WOLLEN': 'want',
    
    # Location prefixes (keep as-is or translate)
    'loc-NORD': 'north-region',
    'loc-SUED': 'south-region',
    'loc-OST': 'east-region',
    'loc-WEST': 'west-region',
    'loc-NORDOST': 'northeast-region',
    'loc-SUEDOST': 'southeast-region',
    'loc-NORDWEST': 'northwest-region',
    'loc-SUEDWEST': 'southwest-region',
    'loc-MITTE': 'central-region',
    'loc-REGION': 'region',
    
    # Special markers (keep or remove)
    '__ON__': '[START]',
    '__OFF__': '[END]',
    '__EMOTION__': '[EMOTION]',
    '__LEFTHAND__': '[LEFT-HAND]',
    '__PU__': '[PAUSE]',
    'ZEIGEN-BILDSCHIRM': 'show-screen',
    'IX': 'point',
}


def translate_gloss(gloss: str) -> str:
    """Translate a single German gloss to English."""
    # Check exact match
    if gloss in GLOSS_TO_ENGLISH:
        return GLOSS_TO_ENGLISH[gloss]
    
    # Check uppercase version
    if gloss.upper() in GLOSS_TO_ENGLISH:
        return GLOSS_TO_ENGLISH[gloss.upper()]
    
    # Check if it's a location prefix
    if gloss.startswith('loc-'):
        base = gloss[4:]  # Remove 'loc-'
        if base in GLOSS_TO_ENGLISH:
            return f"{GLOSS_TO_ENGLISH[base]}-area"
    
    # Return original if no translation found
    return gloss.lower()


def translate_glosses(glosses: list) -> str:
    """Translate a list of German glosses to English sentence."""
    if not glosses:
        return ""
    
    # Translate each gloss
    translated = []
    for gloss in glosses:
        # Skip special markers
        if gloss.startswith('__') or gloss.startswith('<'):
            continue
        
        eng = translate_gloss(gloss)
        if eng and not eng.startswith('['):
            translated.append(eng)
    
    # Join into sentence
    if not translated:
        return ""
    
    # Basic sentence formation
    sentence = ' '.join(translated)
    
    # Capitalize first letter
    sentence = sentence[0].upper() + sentence[1:] if sentence else ""
    
    return sentence


class GlossTranslator:
    """
    Gloss to English translator with optional neural translation fallback.
    """
    
    def __init__(self, use_neural: bool = False):
        self.use_neural = use_neural
        self.neural_model = None
        self.neural_tokenizer = None
        
        if use_neural:
            self._load_neural_model()
    
    def _load_neural_model(self):
        """Load Helsinki-NLP translation model."""
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            model_name = "Helsinki-NLP/opus-mt-de-en"
            print(f"Loading translation model: {model_name}")
            
            self.neural_tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.neural_model = MarianMTModel.from_pretrained(model_name)
            
            # Move to GPU if available
            import torch
            if torch.cuda.is_available():
                self.neural_model = self.neural_model.cuda()
            
            print("Translation model loaded!")
        except ImportError:
            print("Warning: transformers not installed. Using dictionary only.")
            self.use_neural = False
        except Exception as e:
            print(f"Warning: Could not load translation model: {e}")
            self.use_neural = False
    
    def translate(self, glosses: list) -> tuple:
        """
        Translate glosses to English.
        
        Returns:
            (gloss_translation, sentence_translation)
        """
        # Dictionary-based translation (word by word)
        gloss_translation = translate_glosses(glosses)
        
        # Neural translation (if enabled)
        sentence_translation = None
        if self.use_neural and self.neural_model:
            try:
                # Create German sentence from glosses
                german_text = ' '.join([g for g in glosses if not g.startswith('__')])
                
                # Translate
                import torch
                inputs = self.neural_tokenizer(german_text, return_tensors="pt", padding=True)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                translated = self.neural_model.generate(**inputs)
                sentence_translation = self.neural_tokenizer.decode(translated[0], skip_special_tokens=True)
            except Exception as e:
                sentence_translation = f"[Translation error: {e}]"
        
        return gloss_translation, sentence_translation


# Quick test
if __name__ == '__main__':
    test_glosses = ['MORGEN', 'SONNE', 'WARM', 'NORDOST', 'REGEN', 'MOEGLICH']
    
    print("German glosses:", ' '.join(test_glosses))
    print("English translation:", translate_glosses(test_glosses))
    
    # Test with translator class
    translator = GlossTranslator(use_neural=False)
    gloss_trans, _ = translator.translate(test_glosses)
    print("Via translator:", gloss_trans)


