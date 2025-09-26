"""
Create a large synthetic fake news dataset for better model performance
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def create_large_fake_news_dataset(num_samples=1000):
    """Create a large synthetic fake news dataset"""
    
    # Fake news patterns and templates
    fake_templates = [
        "BREAKING: {subject} {action} according to {source}!",
        "SHOCKING: {authority} {reveals} {claim}!",
        "URGENT: {event} {threatens} {target}!",
        "EXCLUSIVE: {celebrity} {caught} in {scandal}!",
        "ALERT: {danger} {causes} {effect} worldwide!",
        "REVEALED: {conspiracy} {controls} {institution}!",
        "CRISIS: {technology} {secretly} {harmful_action}!",
        "EXPOSED: {cover_up} finally {uncovered}!",
        "WARNING: {product} {contains} {harmful_substance}!",
        "SCANDAL: {politician} {admits} {corruption}!"
    ]
    
    # Real news patterns and templates
    real_templates = [
        "{organization} announces {positive_action} for {beneficiary}.",
        "{authority} reports {improvement} in {area} this {timeframe}.",
        "Local {institution} implements new {program} to {benefit}.",
        "{research_body} publishes study on {topic} showing {finding}.",
        "{government_body} approves {policy} for {public_benefit}.",
        "{company} invests in {technology} to improve {service}.",
        "{educational_institution} offers new {course} program for students.",
        "{health_organization} recommends {health_measure} for {season}.",
        "{environmental_group} launches {initiative} to protect {resource}.",
        "{community_group} organizes {event} to support {cause}."
    ]
    
    # Word banks for templates
    subjects = ["Scientists", "Researchers", "Experts", "Government officials", "Military", "Intelligence agencies"]
    actions = ["discover", "reveal", "uncover", "expose", "confirm", "prove", "find evidence of"]
    sources = ["leaked documents", "whistleblower reports", "classified files", "secret recordings", "insider sources"]
    
    authorities = ["Government", "CIA", "FBI", "Military", "Big Pharma", "Tech giants", "Media"]
    reveals = ["admits", "confirms", "reveals", "exposes", "acknowledges"]
    claims = ["mind control experiments", "population control plans", "secret surveillance", "hidden agendas"]
    
    events = ["Solar flares", "Asteroid impact", "Economic collapse", "Cyber attacks", "Pandemic outbreak"]
    threatens = ["will destroy", "threatens to eliminate", "could wipe out", "plans to target"]
    targets = ["major cities", "the internet", "global economy", "human civilization", "democratic institutions"]
    
    celebrities = ["Hollywood star", "Famous politician", "Tech billionaire", "Royal family member", "Pop star"]
    caught = ["caught", "filmed", "recorded", "photographed", "witnessed"]
    scandals = ["massive corruption scandal", "secret affair", "illegal activities", "tax evasion scheme"]
    
    dangers = ["5G towers", "Vaccines", "Chemtrails", "GMO foods", "Artificial intelligence"]
    causes = ["cause", "trigger", "lead to", "result in", "create"]
    effects = ["mass extinction", "brain damage", "genetic mutations", "mind control", "infertility"]
    
    conspiracies = ["Illuminati", "Secret societies", "Alien overlords", "Shadow government", "Reptilian elite"]
    controls = ["secretly controls", "manipulates", "influences", "dominates", "rules"]
    institutions = ["world governments", "global economy", "media outlets", "educational systems"]
    
    technologies = ["Smartphones", "Social media", "Smart TVs", "Internet", "Satellites"]
    secretly = ["secretly", "covertly", "invisibly", "silently", "unknowingly"]
    harmful_actions = ["monitor thoughts", "collect DNA", "spread propaganda", "control behavior"]
    
    cover_ups = ["Moon landing hoax", "JFK assassination truth", "Area 51 secrets", "9/11 inside job"]
    uncovered = ["surfaces", "emerges", "comes to light", "gets exposed", "is revealed"]
    
    products = ["Popular soda", "Common medication", "Household cleaner", "Beauty product", "Food additive"]
    contains = ["contains", "includes", "has", "is made with", "is laced with"]
    harmful_substances = ["mind control chemicals", "cancer-causing agents", "toxic metals", "addictive compounds"]
    
    politicians = ["Senator", "Governor", "Mayor", "Congressman", "President"]
    admits = ["admits to", "confesses", "acknowledges", "reveals", "discloses"]
    corruption = ["taking bribes", "election fraud", "money laundering", "insider trading"]
    
    # Real news word banks
    organizations = ["City Council", "Health Department", "University", "Hospital", "Library System", "School District"]
    positive_actions = ["funding increase", "new program launch", "service expansion", "facility upgrade"]
    beneficiaries = ["local residents", "students", "patients", "community members", "families"]
    
    authorities_real = ["Department of Health", "Transportation Authority", "Environmental Agency", "Education Board"]
    improvements = ["significant improvement", "positive trends", "steady progress", "notable advances"]
    areas = ["air quality", "public safety", "education outcomes", "healthcare services", "infrastructure"]
    timeframes = ["quarter", "month", "year", "season", "period"]
    
    institutions = ["community center", "hospital", "school", "library", "park system"]
    programs = ["wellness program", "education initiative", "safety protocol", "support service"]
    benefits = ["improve accessibility", "enhance safety", "support families", "boost economy"]
    
    research_bodies = ["University researchers", "Medical center", "Research institute", "Scientific team"]
    topics = ["renewable energy", "public health", "environmental protection", "education methods"]
    findings = ["positive results", "promising outcomes", "significant benefits", "important insights"]
    
    government_bodies = ["State legislature", "City council", "County board", "Municipal authority"]
    policies = ["new legislation", "funding bill", "safety regulation", "support program"]
    public_benefits = ["public safety", "environmental protection", "economic development", "social welfare"]
    
    companies = ["Local business", "Technology company", "Healthcare provider", "Energy company"]
    technologies_real = ["renewable energy", "digital infrastructure", "medical equipment", "transportation systems"]
    services = ["customer service", "healthcare delivery", "educational access", "public transportation"]
    
    educational_institutions = ["Community college", "University", "Technical school", "Training center"]
    courses = ["professional development", "technical training", "continuing education", "skill building"]
    
    health_organizations = ["Health department", "Medical association", "Wellness center", "Public health office"]
    health_measures = ["vaccination programs", "health screenings", "wellness initiatives", "safety protocols"]
    seasons = ["flu season", "summer months", "winter period", "allergy season"]
    
    environmental_groups = ["Conservation society", "Environmental group", "Green initiative", "Sustainability organization"]
    initiatives = ["cleanup campaign", "conservation program", "awareness drive", "protection effort"]
    resources = ["local waterways", "natural habitats", "air quality", "green spaces"]
    
    community_groups = ["Volunteer organization", "Civic group", "Neighborhood association", "Community foundation"]
    events = ["fundraising event", "awareness campaign", "volunteer drive", "community gathering"]
    causes = ["local charities", "education funding", "healthcare access", "environmental protection"]
    
    # Generate fake news
    fake_news = []
    for _ in range(num_samples // 2):
        template = random.choice(fake_templates)
        
        # Fill template with random words
        if "{subject}" in template:
            text = template.format(
                subject=random.choice(subjects),
                action=random.choice(actions),
                source=random.choice(sources)
            )
        elif "{authority}" in template:
            text = template.format(
                authority=random.choice(authorities),
                reveals=random.choice(reveals),
                claim=random.choice(claims)
            )
        elif "{event}" in template:
            text = template.format(
                event=random.choice(events),
                threatens=random.choice(threatens),
                target=random.choice(targets)
            )
        elif "{celebrity}" in template:
            text = template.format(
                celebrity=random.choice(celebrities),
                caught=random.choice(caught),
                scandal=random.choice(scandals)
            )
        elif "{danger}" in template:
            text = template.format(
                danger=random.choice(dangers),
                causes=random.choice(causes),
                effect=random.choice(effects)
            )
        elif "{conspiracy}" in template:
            text = template.format(
                conspiracy=random.choice(conspiracies),
                controls=random.choice(controls),
                institution=random.choice(institutions)
            )
        elif "{technology}" in template:
            text = template.format(
                technology=random.choice(technologies),
                secretly=random.choice(secretly),
                harmful_action=random.choice(harmful_actions)
            )
        elif "{cover_up}" in template:
            text = template.format(
                cover_up=random.choice(cover_ups),
                uncovered=random.choice(uncovered)
            )
        elif "{product}" in template:
            text = template.format(
                product=random.choice(products),
                contains=random.choice(contains),
                harmful_substance=random.choice(harmful_substances)
            )
        elif "{politician}" in template:
            text = template.format(
                politician=random.choice(politicians),
                admits=random.choice(admits),
                corruption=random.choice(corruption)
            )
        else:
            text = template
            
        fake_news.append(text)
    
    # Generate real news
    real_news = []
    for _ in range(num_samples // 2):
        template = random.choice(real_templates)
        
        # Fill template with random words
        if "{organization}" in template:
            text = template.format(
                organization=random.choice(organizations),
                positive_action=random.choice(positive_actions),
                beneficiary=random.choice(beneficiaries)
            )
        elif "{authority}" in template:
            text = template.format(
                authority=random.choice(authorities_real),
                improvement=random.choice(improvements),
                area=random.choice(areas),
                timeframe=random.choice(timeframes)
            )
        elif "{institution}" in template:
            text = template.format(
                institution=random.choice(institutions),
                program=random.choice(programs),
                benefit=random.choice(benefits)
            )
        elif "{research_body}" in template:
            text = template.format(
                research_body=random.choice(research_bodies),
                topic=random.choice(topics),
                finding=random.choice(findings)
            )
        elif "{government_body}" in template:
            text = template.format(
                government_body=random.choice(government_bodies),
                policy=random.choice(policies),
                public_benefit=random.choice(public_benefits)
            )
        elif "{company}" in template:
            text = template.format(
                company=random.choice(companies),
                technology=random.choice(technologies_real),
                service=random.choice(services)
            )
        elif "{educational_institution}" in template:
            text = template.format(
                educational_institution=random.choice(educational_institutions),
                course=random.choice(courses)
            )
        elif "{health_organization}" in template:
            text = template.format(
                health_organization=random.choice(health_organizations),
                health_measure=random.choice(health_measures),
                season=random.choice(seasons)
            )
        elif "{environmental_group}" in template:
            text = template.format(
                environmental_group=random.choice(environmental_groups),
                initiative=random.choice(initiatives),
                resource=random.choice(resources)
            )
        elif "{community_group}" in template:
            text = template.format(
                community_group=random.choice(community_groups),
                event=random.choice(events),
                cause=random.choice(causes)
            )
        else:
            text = template
            
        real_news.append(text)
    
    # Combine and create DataFrame
    all_texts = fake_news + real_news
    all_labels = [1] * len(fake_news) + [0] * len(real_news)  # 1 for fake, 0 for real
    
    # Shuffle the data
    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': all_texts,
        'label': all_labels
    })
    
    return df

if __name__ == "__main__":
    print("🔄 Generating large synthetic fake news dataset...")
    
    # Create dataset with 1000 samples
    df = create_large_fake_news_dataset(1000)
    
    # Save to CSV
    df.to_csv('data/large_fake_news_dataset.csv', index=False)
    
    print(f"✅ Dataset created with {len(df)} samples")
    print(f"📊 Fake news: {sum(df['label'])} samples")
    print(f"📊 Real news: {len(df) - sum(df['label'])} samples")
    print("💾 Saved to data/large_fake_news_dataset.csv")