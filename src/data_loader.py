"""
Data Loader Module
Handles loading and initial exploration of fake news datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

class DataLoader:
    def __init__(self):
        self.data_path = "data/"
        
    def load_data(self, file_path=None):
        """
        Load fake news dataset
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            DataFrame with text and label columns
        """
        print("📊 Loading fake news dataset...")
        
        if file_path and os.path.exists(file_path):
            print(f"📁 Loading dataset from: {file_path}")
            df = pd.read_csv(file_path)
        elif os.path.exists('data/large_fake_news_dataset.csv'):
            print("📁 Loading large synthetic dataset...")
            df = pd.read_csv('data/large_fake_news_dataset.csv')
        elif os.path.exists('data/fake_news_dataset.csv'):
            print("📁 Loading existing dataset from data/fake_news_dataset.csv")
            df = pd.read_csv('data/fake_news_dataset.csv')
        else:
            print("📝 No existing dataset found. Creating sample dataset...")
            df = self._create_sample_dataset()
            
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Total samples: {len(df)}")
        print(f"📊 Fake news: {sum(df['label'])} samples")
        print(f"📊 Real news: {len(df) - sum(df['label'])} samples")
        
        return df
    
    def _create_sample_dataset(self):
        """
        Create a comprehensive sample fake news dataset for demonstration
        """
        print("📝 Creating comprehensive sample fake news dataset...")
        
        # Expanded real news data with credible reporting patterns
        real_news = [
            "Scientists discover new species of marine life in deep ocean trenches during recent expedition",
            "Local government announces new infrastructure project to improve city transportation",
            "Research shows positive effects of renewable energy adoption on local economies",
            "University researchers publish findings on climate change impact on agriculture",
            "New medical breakthrough offers hope for patients with rare genetic disorders",
            "Technology company reports quarterly earnings exceeding market expectations",
            "International summit addresses global cooperation on environmental issues",
            "Educational reforms aim to improve student outcomes in underserved communities",
            "Archaeological team uncovers ancient artifacts providing insights into historical civilization",
            "Public health officials recommend updated vaccination guidelines for seasonal flu",
            "Local community center opens new programs for senior citizens this fall",
            "City council approves budget for infrastructure improvements next year",
            "University researchers publish findings on renewable energy efficiency",
            "New public transportation route connects suburban areas to downtown",
            "Local hospital receives accreditation for patient safety standards",
            "School district implements new technology program for students",
            "Environmental group organizes cleanup event at local park this weekend",
            "Small business association offers workshops for entrepreneurs",
            "Library system expands digital resources and online services",
            "Municipal water treatment facility undergoes scheduled maintenance",
            "Regional unemployment rate drops to lowest level in five years",
            "State legislature passes bill to improve highway safety measures",
            "Medical center announces expansion of emergency services department",
            "Public health officials recommend seasonal flu vaccination campaign",
            "Agricultural department reports successful harvest season for local farmers",
            "Education board approves new curriculum standards for mathematics",
            "Police department launches community outreach program in neighborhoods",
            "Fire department receives federal grant for new equipment purchase",
            "Parks and recreation opens registration for summer youth programs",
            "Housing authority announces affordable housing development project"
        ]
        
        # Expanded fake news data with various patterns
        fake_news = [
            "SHOCKING: Aliens secretly control world governments according to leaked documents",
            "BREAKING: Miracle cure for all diseases discovered but hidden by pharmaceutical companies",
            "EXCLUSIVE: Celebrity caught in scandal that will destroy their career forever",
            "URGENT: New law will ban all social media platforms starting next month",
            "REVEALED: Secret government plan to control population through water supply",
            "AMAZING: Local man discovers simple trick that doctors hate for instant weight loss",
            "CONSPIRACY: Major news outlets spreading lies about recent election results",
            "DANGER: Common household item causes cancer according to suppressed study",
            "SCANDAL: Politician admits to taking bribes from foreign governments in secret recording",
            "ALERT: Natural disaster predicted to destroy major city within days",
            "BREAKING: Scientists discover aliens living among us in secret underground cities!",
            "SHOCKING: Government hiding cure for all diseases to control population!",
            "URGENT: New study proves vaccines contain mind control chips!",
            "EXCLUSIVE: Celebrity caught in massive scandal that will shock you!",
            "ALERT: Economic collapse imminent, withdraw all your money now!",
            "REVEALED: Secret society controls all world governments!",
            "BREAKING: Miracle diet pill melts fat overnight without exercise!",
            "SHOCKING: Weather machines causing natural disasters worldwide!",
            "URGENT: Social media platforms reading your thoughts!",
            "EXCLUSIVE: Time travel technology secretly developed by military!",
            "BOMBSHELL: Ancient pyramid found on Mars by NASA rover!",
            "CRISIS: 5G towers causing mass bird deaths worldwide!",
            "EXPOSED: Flat Earth society proves globe theory is hoax!",
            "WARNING: Chemtrails poisoning water supply across nation!",
            "SCANDAL: Politicians secretly replaced by robot doubles!",
            "URGENT: Asteroid heading toward Earth, government covers up!",
            "SHOCKING: Bigfoot captured on camera in national park!",
            "BREAKING: Illuminati meeting leaked, world domination plan revealed!",
            "ALERT: Fake moon landing evidence finally surfaces after decades!",
            "EXCLUSIVE: Reptilian shapeshifters infiltrate world leaders!"
        ]
        
        # Create DataFrame
        texts = real_news + fake_news
        labels = [0] * len(real_news) + [1] * len(fake_news)  # 0 = real, 1 = fake
        
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        # Save sample dataset
        os.makedirs(self.data_path, exist_ok=True)
        df.to_csv(os.path.join(self.data_path, "fake_news.csv"), index=False)
        
        return df
    
    def perform_eda(self, df):
        """
        Perform Exploratory Data Analysis
        """
        print("\n📊 EXPLORATORY DATA ANALYSIS")
        print("=" * 40)
        
        # Basic statistics
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Label distribution
        label_counts = df['label'].value_counts()
        print(f"\nLabel Distribution:")
        print(f"Real News (0): {label_counts[0]} ({label_counts[0]/len(df)*100:.1f}%)")
        print(f"Fake News (1): {label_counts[1]} ({label_counts[1]/len(df)*100:.1f}%)")
        
        # Text length analysis
        df['text_length'] = df['text'].str.len()
        print(f"\nText Length Statistics:")
        print(f"Average length: {df['text_length'].mean():.1f} characters")
        print(f"Min length: {df['text_length'].min()} characters")
        print(f"Max length: {df['text_length'].max()} characters")
        
        # Create visualizations
        self._create_eda_plots(df)
        
    def _create_eda_plots(self, df):
        """
        Create EDA visualizations
        """
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Label distribution
        df['label'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'salmon'])
        axes[0,0].set_title('Distribution of Real vs Fake News')
        axes[0,0].set_xlabel('Label (0=Real, 1=Fake)')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=0)
        
        # Text length distribution
        axes[0,1].hist(df[df['label']==0]['text_length'], alpha=0.7, label='Real', bins=10, color='skyblue')
        axes[0,1].hist(df[df['label']==1]['text_length'], alpha=0.7, label='Fake', bins=10, color='salmon')
        axes[0,1].set_title('Text Length Distribution')
        axes[0,1].set_xlabel('Text Length (characters)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # Word clouds
        real_text = ' '.join(df[df['label']==0]['text'])
        fake_text = ' '.join(df[df['label']==1]['text'])
        
        # Real news word cloud
        if len(real_text) > 0:
            try:
                wordcloud_real = WordCloud(
                    width=300, height=200, 
                    background_color='white',
                    font_path=None,
                    prefer_horizontal=0.9
                ).generate(real_text)
                axes[1,0].imshow(wordcloud_real, interpolation='bilinear')
                axes[1,0].set_title('Real News Word Cloud')
                axes[1,0].axis('off')
            except Exception as e:
                axes[1,0].text(0.5, 0.5, 'WordCloud\nNot Available', 
                              ha='center', va='center', transform=axes[1,0].transAxes)
                axes[1,0].set_title('Real News Word Cloud')
                axes[1,0].axis('off')
        
        # Fake news word cloud
        if len(fake_text) > 0:
            try:
                wordcloud_fake = WordCloud(
                    width=300, height=200, 
                    background_color='white',
                    font_path=None,
                    prefer_horizontal=0.9
                ).generate(fake_text)
                axes[1,1].imshow(wordcloud_fake, interpolation='bilinear')
                axes[1,1].set_title('Fake News Word Cloud')
                axes[1,1].axis('off')
            except Exception as e:
                axes[1,1].text(0.5, 0.5, 'WordCloud\nNot Available', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Fake News Word Cloud')
                axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 EDA plots saved to results/eda_analysis.png")