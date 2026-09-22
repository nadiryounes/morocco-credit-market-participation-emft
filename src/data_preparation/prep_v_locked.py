import re
import pandas as pd
import json
import os

# Update BIB file for Mol 2019 and add Maldonado 2021
bib_file = "REFERENCES_VERIFIED_V2.bib"
if os.path.exists(bib_file):
    with open(bib_file, "r") as f:
        bib_content = f.read()
    
    # Fix Mol et al.
    bib_content = bib_content.replace("mol2018", "mol2019")
    bib_content = re.sub(r'year=\{2018\}', 'year={2019}', bib_content, count=1)
    
    # Check if Maldonado is there, if not add it
    if "maldonado2021" not in bib_content.lower():
        maldonado = """
@article{maldonado2021,
  title={Time-weighted Fuzzy Support Vector Machines for classification in changing environments},
  author={Maldonado, Sebasti{\\'a}n and L{\\'o}pez, Julio and Vairetti, Cristi{\\'a}n},
  journal={Information Sciences},
  volume={559},
  pages={97--110},
  year={2021},
  doi={10.1016/j.ins.2021.01.070}
}
"""
        bib_content += maldonado

    with open(bib_file, "w") as f:
        f.write(bib_content)

intro_locked = """# 1. Introduction

Access to external finance remains a fundamental determinant of firm survival and investment capacity. However, measuring financial access strictly through observed loan volumes fails to capture the full spectrum of latent credit demand and firm participation decisions. A critical distinction must be maintained between unobservable credit demand, the observable decision to submit a formal loan application, and the broader phenomenon of constrained non-application. While some non-applicants abstain from the credit market simply because they possess no financing need, others are constrained by credit-market frictions (Cole and Sokolyk, 2016). These constrained non-applicants include those facing strict expected-rejection discouragement (Kon and Storey, 2003), as well as firms deterred by complex application procedures, collateral requirements, or unfavorable interest rates. Failing to distinguish among these distinct credit-market states risks conflating unconstrained firms with those experiencing severe access frictions, thereby distorting economic research.

Evaluating these credit-market states through cross-sectional averages often conceals the true extent of firm-level transitions. As highlighted by the dynamic financing literature (Cowling and Sclip, 2023), a relatively similar aggregate state distribution across two distinct time periods does not imply that individual firms remain in the same financing state. Firms may continuously enter and exit states of constraint, active application, and sufficient internal liquidity. Relying solely on aggregate point-in-time compositions provides an incomplete understanding of structural credit-market mobility.

The period spanning the COVID-19 pandemic provides an informative context for analyzing these longitudinal financing dynamics. In the 2019 baseline, domestic credit to the private sector in Morocco stood at 62.7% of GDP. The subsequent period of macroeconomic disruption introduced operational and liquidity shocks, prompting policy interventions such as the Damane Oxygene guaranteed loan program, which benefited 49,360 enterprises by 16 October 2020. An empirical architecture spanning a 2019 baseline wave, COVID-era follow-ups, and a 2023 wave offers an observation window of firm financing trajectories. Within this specific context, the COVID-era follow-ups provide intermediate measures of firm financial stress without causal identification.

Beyond descriptive transitions and inferential associations, changes in the macroeconomic environment pose a predictive validity problem for financial classification models. Models that learn the characteristics of constrained firms in a specific 2019 baseline environment may not reliably maintain their performance when transported to the 2023 wave. A central challenge in financial machine learning is temporal distribution shift, wherein the relationships between firm characteristics and credit-market states drift as the broader economic environment changes (Dal Pozzolo et al., 2015). Dataset shift and the temporal mismatch between training and deployment periods highlight the necessity of strict out-of-time evaluation in credit scoring (Maldonado et al., 2021). Distinguishing random internal cross-validation from true out-of-time temporal validation is therefore critical to understanding the limits of predictive algorithms deployed across a period of macroeconomic disruption.

To our knowledge, existing studies have not combined in one Moroccan firm-level research design: repeated weighted credit-state distributions; matched 2019-2023 firm transitions; intermediate COVID-era follow-up observations; and held-out temporal transportability testing of credit-state classifiers. The closest competitor literature has yielded critical insights into borrower discouragement, longitudinal firm dynamics, credit constraints facing Arab and Moroccan SMEs, and the vulnerability of financial classifiers to temporal shift. The study combines repeated cross-sectional estimation and matched-firm transition analysis with a separate strictly held-out temporal evaluation of predictive models.

The overarching research question of this paper asks how firm participation in the credit market evolves when aggregate credit-state composition, individual firm transitions, intermediate COVID-era financial conditions, and the temporal validity of predictive relationships are examined jointly. To address this, the study makes three contributions. First, the study jointly compares aggregate credit-state composition with within-firm transitions. Second, it examines associations between intermediate COVID-era financial-stress measures and 2023 constrained non-application. Third, it tests temporal transportability from 2019 to a fully held-out 2023 sample.

The remainder of this paper is structured as follows. Section 2 outlines the conceptual background, reviews the relevant literature, and formalizes the research questions. Section 3 details the data sources and empirical methodology. Section 4 presents the quantitative results. Section 5 discusses the implications, and Section 6 concludes.
"""

lit_locked = """# 2. Literature Review and Conceptual Background

## 2.1 Credit rationing and information asymmetry
The theoretical foundation of credit-market frictions rests heavily on the economics of information asymmetry. Stiglitz and Weiss (1981) demonstrate that because lenders possess imperfect information regarding the true risk profiles and project qualities of prospective borrowers, the standard market-clearing mechanism of price adjustment fails. Specifically, lenders cannot simply raise interest rates to clear excess credit demand without inducing adverse selection. Higher interest rates disproportionately attract high-risk borrowers who are willing to accept onerous terms because they have a lower probability of repaying the principal. Furthermore, higher rates create incentive effects by encouraging borrowers to undertake riskier projects with higher potential payoffs but higher probabilities of default. Consequently, profit-maximizing lenders may engage in equilibrium credit rationing, effectively denying loans to observationally indistinguishable firms even if those firms are willing to pay higher rates. Because of this structural rationing, observed loan volumes do not fully identify latent credit demand.

## 2.2 Constrained non-application and borrower discouragement
While traditional credit rationing frameworks focus on firms that actively apply for loans and are rejected, subsequent literature identifies a subset of firms that bypass the application process entirely. Kon and Storey (2003) formulated a theory of borrower discouragement wherein good borrowers may choose not to apply due to expected rejection, application costs, and bank screening errors. In our empirical application, the survey variable $k17 == 6$ captures the expected-rejection manifestation of discouragement in the survey.

However, Cole and Sokolyk (2016) emphasize the critical empirical importance of distinguishing between unconstrained non-applicants—firms with no financing need—and firms that face genuine credit-market frictions. Our study operationalizes a broader category of constrained non-application that encompasses strict expected-rejection discouragement but also includes firms deterred by complex application procedures, high collateral requirements, or unfavorable interest rates. Recent empirical work underscores how macroeconomic and institutional factors influence these dimensions. Mol-Gómez-Vázquez et al. (2019) find that borrower discouragement decreases with bank market power, presenting nonlinear and context-dependent patterns in less-developed or high-market-power environments. Aristei et al. (2024) find that less financially literate entrepreneurs are more likely to be discouraged from applying, particularly in relation to higher application costs. Furthermore, Mc Namara et al. (2025) document that borrower discouragement varies with multiple dimensions of national lending infrastructure, including the credit information, legal, judicial, bankruptcy, tax, and regulatory environment.

## 2.3 Dynamic credit-market participation
Credit-market participation is increasingly recognized as a dynamic state rather than a fixed firm characteristic. Utilizing a cross-country panel of firms re-surveyed across successive SAFE waves, Cowling and Sclip (2023) emphasize the concept of dynamic transitions into and out of discouragement. They document that these transitions exhibit significant associations with business-cycle variation, past credit experiences, risk indicators, credit history, and profit outlook. This dynamic perspective demonstrates that cross-sectional analyses of borrower discouragement or constraint provide only a point-in-time snapshot. Because aggregate participation rates can remain theoretically stable even while individual firms experience mobility across credit states, longitudinal tracking is essential. Comparing aggregate cross-sectional transitions against matched-firm transitions provides a more precise mechanism for understanding mobility within the credit market.

## 2.4 SME finance in Morocco and comparable emerging markets
Within the Middle East and North Africa (MENA) region, the institutional environment limits formal credit access for SMEs. Gourène et al. (2025) examine family-owned SMEs in Egypt, Jordan, Morocco, and Tunisia, finding that these firms have greater credit needs but are less likely to apply and are therefore more credit constrained. They also document that governance, formal strategy, and managerial practices are associated with easier credit access in these markets. 

In the specific context of Morocco, Boutfssi and Quamar (2024a, 2024b) investigate the determinants of SME credit rationing within the Casablanca-Settat region, highlighting local constraints and procedural complexities. While pre-pandemic domestic credit to the private sector in Morocco reached approximately 62.7% of GDP in 2019 (World Bank), Boutfssi and Quamar (2024a) confirm that substantial segments of the SME sector still experienced structural rationing. During the 2019-2023 study period, the Moroccan institutional lending infrastructure experienced changes; for instance, verified institutional sources indicate that Morocco launched a national electronic movable-collateral registry in early 2020. Additionally, the COVID-era Damane Oxygene guaranteed loan mechanism was introduced, providing state-backed liquidity support to 49,360 enterprises by 16 October 2020. Against this backdrop, analyzing Moroccan SME credit constraints provides a highly relevant emerging-market setting to evaluate longitudinal firm behavior over a period of macroeconomic disruption.

## 2.5 Temporal validity of predictive financial models
The expansion of machine learning and predictive analytics in corporate finance has introduced tools for classifying firm credit states. However, methodological literature emphasizes a distinction between internal cross-validation performance and true out-of-time transportability. Hand (2006) outlines general classifier-evaluation caution, warning against the illusion of progress when models are assessed in static test environments rather than in evolving real-world conditions. 

Maldonado et al. (2021) document dataset shift in credit scoring, emphasizing the temporal mismatch between training and deployment periods and highlighting the critical necessity of out-of-time evaluation. In the domain of financial concept drift, Dal Pozzolo et al. (2015) similarly document how predictive relationships change over time within the context of credit card fraud detection. Furthermore, Lessmann et al. (2015) provide comprehensive benchmarking of state-of-the-art classification algorithms for credit scoring, establishing baseline best practices for model evaluation. Consequently, evaluating the temporal transportability of a credit-state classifier requires fully holding out a subsequent temporal sample to test the model's predictive validity across a period of macroeconomic disruption.

## 2.6 Research questions and analytical expectations
To investigate the joint dynamics of credit-market composition, intermediate financial stress, and temporal predictive validity, the empirical design is organized around four research questions:

RQ1: How do weighted aggregate credit-market state distributions differ between the 2019 baseline and the 2023 wave?

RQ2: How do individual firms transition among credit-market states between 2019 and 2023?

RQ3: Are cumulative COVID-era arrears and liquidity stress associated with constrained non-application in 2023, conditional on baseline credit-market state?

RQ4: How well do credit-state classifiers developed on the 2019 sample transport to the held-out 2023 sample?

For the intermediate financial stress analysis (RQ3), the two COVID-era exposures constituted the prespecified confirmatory exposure family, evaluated with Holm adjustment for multiple testing.
"""

with open("MANUSCRIPT_INTRODUCTION_LOCKED.md", "w") as f:
    f.write(intro_locked)

with open("MANUSCRIPT_LITERATURE_REVIEW_LOCKED.md", "w") as f:
    f.write(lit_locked)

trace = [
    {"SECTION": "1", "PARAGRAPH_ID": "P1", "CLAIM": "Some non-applicants abstain due to no financing need, others are constrained.", "CITATION_KEY": "cole2016", "EXACT_SOURCE_FINDING": "Importance of distinguishing unconstrained from supply-side constrained firms.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Cited in Intro."},
    {"SECTION": "1", "PARAGRAPH_ID": "P1", "CLAIM": "Strict expected-rejection discouragement.", "CITATION_KEY": "kon2003", "EXACT_SOURCE_FINDING": "Formalizes theory of discouragement.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Cited in Intro."},
    {"SECTION": "1", "PARAGRAPH_ID": "P2", "CLAIM": "Dynamic financing literature highlights transitions.", "CITATION_KEY": "cowling2023", "EXACT_SOURCE_FINDING": "Dynamic discouraged borrowers transition over time.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Cited in Intro."},
    {"SECTION": "1", "PARAGRAPH_ID": "P4", "CLAIM": "Concept drift in broader economic environment changes.", "CITATION_KEY": "dalpozzolo2015", "EXACT_SOURCE_FINDING": "Temporal distribution shift and concept drift.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Cited in Intro."},
    {"SECTION": "1", "PARAGRAPH_ID": "P4", "CLAIM": "Dataset shift and temporal mismatch in credit scoring.", "CITATION_KEY": "maldonado2021", "EXACT_SOURCE_FINDING": "Dataset shift in credit scoring environments.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Cited in Intro."},
    {"SECTION": "2.1", "PARAGRAPH_ID": "P1", "CLAIM": "Information asymmetry leads to adverse selection, incentive effects, and equilibrium rationing.", "CITATION_KEY": "stiglitz1981", "EXACT_SOURCE_FINDING": "Stiglitz and Weiss on imperfect information and credit rationing.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.1."},
    {"SECTION": "2.2", "PARAGRAPH_ID": "P1", "CLAIM": "Good borrowers discouraged by expected rejection, costs, and screening errors.", "CITATION_KEY": "kon2003", "EXACT_SOURCE_FINDING": "Theory of borrower discouragement.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.2."},
    {"SECTION": "2.2", "PARAGRAPH_ID": "P2", "CLAIM": "Distinguishing unconstrained non-applicants from genuine frictions.", "CITATION_KEY": "cole2016", "EXACT_SOURCE_FINDING": "Distinguishes unconstrained non-applicants.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.2."},
    {"SECTION": "2.2", "PARAGRAPH_ID": "P2", "CLAIM": "Borrower discouragement decreases with bank market power.", "CITATION_KEY": "mol2019", "EXACT_SOURCE_FINDING": "Discouragement decreases with bank power with nonlinear patterns.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.2."},
    {"SECTION": "2.2", "PARAGRAPH_ID": "P2", "CLAIM": "Financial literacy mitigates self-rationing (application costs).", "CITATION_KEY": "aristei2024", "EXACT_SOURCE_FINDING": "Less financially literate entrepreneurs are more discouraged due to application costs.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.2."},
    {"SECTION": "2.2", "PARAGRAPH_ID": "P2", "CLAIM": "Discouragement varies with national lending infrastructure.", "CITATION_KEY": "mcnamara2025", "EXACT_SOURCE_FINDING": "Discouragement varies with credit information, legal, judicial, bankruptcy, tax.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.2."},
    {"SECTION": "2.3", "PARAGRAPH_ID": "P1", "CLAIM": "Transitions in/out of discouragement (SAFE panel).", "CITATION_KEY": "cowling2023", "EXACT_SOURCE_FINDING": "Cross-country SAFE panel shows dynamic transitions linked to business cycle and risk.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.3."},
    {"SECTION": "2.4", "PARAGRAPH_ID": "P1", "CLAIM": "Family-owned SMEs in MENA have greater credit needs but apply less.", "CITATION_KEY": "gourene2025", "EXACT_SOURCE_FINDING": "SMEs in Egypt, Jordan, Morocco, Tunisia have greater needs, apply less.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.4."},
    {"SECTION": "2.4", "PARAGRAPH_ID": "P2", "CLAIM": "Determinants of SME credit rationing in Casablanca-Settat.", "CITATION_KEY": "boutfssi2024a", "EXACT_SOURCE_FINDING": "Investigates determinants of SME credit rationing.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.4."},
    {"SECTION": "2.4", "PARAGRAPH_ID": "P2", "CLAIM": "Determinants of SME credit rationing in Casablanca-Settat.", "CITATION_KEY": "boutfssi2024b", "EXACT_SOURCE_FINDING": "Investigates determinants of SME credit rationing.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.4."},
    {"SECTION": "2.5", "PARAGRAPH_ID": "P1", "CLAIM": "General classifier-evaluation caution, illusion of progress.", "CITATION_KEY": "hand2006", "EXACT_SOURCE_FINDING": "Illusion of progress when models are assessed statically.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.5."},
    {"SECTION": "2.5", "PARAGRAPH_ID": "P2", "CLAIM": "Dataset shift in credit scoring.", "CITATION_KEY": "maldonado2021", "EXACT_SOURCE_FINDING": "Dataset shift and temporal mismatch in credit scoring.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.5."},
    {"SECTION": "2.5", "PARAGRAPH_ID": "P2", "CLAIM": "Concept drift in financial fraud context.", "CITATION_KEY": "dalpozzolo2015", "EXACT_SOURCE_FINDING": "Concept drift in credit card fraud detection.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.5."},
    {"SECTION": "2.5", "PARAGRAPH_ID": "P2", "CLAIM": "Credit scoring benchmarking.", "CITATION_KEY": "lessmann2015", "EXACT_SOURCE_FINDING": "Comprehensive benchmarking of credit scoring algorithms.", "SUPPORTED_FULLY_YN": "YES", "QUALIFICATION_NEEDED": "NONE", "ACTION_TAKEN": "Section 2.5."}
]

audit = [
    {"TOPIC": "finance-growth motivation", "CLAIM": "Access to external finance remains a fundamental determinant of firm survival and investment capacity.", "CLASSIFICATION": "SUPPORTED", "NOTES": "Standard corporate finance motivation."},
    {"TOPIC": "credit-state distinction", "CLAIM": "Unobservable credit demand vs. formal application vs. constrained non-application.", "CLASSIFICATION": "SUPPORTED", "NOTES": "Supported by Cole & Sokolyk and Kon & Storey."},
    {"TOPIC": "COVID/Morocco context", "CLAIM": "62.7% domestic credit/GDP in 2019 baseline.", "CLASSIFICATION": "SUPPORTED", "NOTES": "Verified in MOROCCO_FINANCE_CONTEXT_V2.csv."},
    {"TOPIC": "Damane Oxygene", "CLAIM": "49,360 enterprises by 16 October 2020.", "CLASSIFICATION": "SUPPORTED", "NOTES": "Exact verified institutional source formulation."},
    {"TOPIC": "collateral registry", "CLAIM": "Morocco launched a national electronic movable-collateral registry in early 2020.", "CLASSIFICATION": "SUPPORTED", "NOTES": "Exact verified finding."},
    {"TOPIC": "novelty statement", "CLAIM": "The study combines repeated cross-sectional estimation and matched-firm transition analysis with a separate strictly held-out temporal evaluation of predictive models.", "CLASSIFICATION": "SUPPORTED", "NOTES": "Restated as combination rather than claiming equivalent standards."},
    {"TOPIC": "credit-rationing theory", "CLAIM": "Adverse selection, incentive effects, and equilibrium rationing.", "CLASSIFICATION": "SUPPORTED", "NOTES": "Direct from Stiglitz and Weiss (1981)."},
    {"TOPIC": "discouragement definition", "CLAIM": "Expected rejection (k17==6) maps to Kon & Storey strict theory.", "CLASSIFICATION": "SUPPORTED", "NOTES": "Properly distinguishes survey variable from general theory."},
    {"TOPIC": "MENA evidence", "CLAIM": "Family-owned SMEs in Egypt, Jordan, Morocco, Tunisia have greater credit needs but are less likely to apply.", "CLASSIFICATION": "SUPPORTED", "NOTES": "Direct finding from Gourene et al. (2025)."},
    {"TOPIC": "temporal ML validity", "CLAIM": "Dataset shift in credit scoring (Maldonado); concept drift in fraud (Dal Pozzolo); classifier caution (Hand).", "CLASSIFICATION": "SUPPORTED", "NOTES": "Maldonado (2021) added as direct out-of-time credit scoring source."},
    {"TOPIC": "contribution 1", "CLAIM": "Jointly compares aggregate credit-state composition with within-firm transitions.", "CLASSIFICATION": "SUPPORTED", "NOTES": "Neutral wording, no result leakage."},
    {"TOPIC": "contribution 2", "CLAIM": "Examines associations between intermediate COVID-era financial-stress measures and 2023 constrained non-application.", "CLASSIFICATION": "SUPPORTED", "NOTES": "Neutral wording, no causal inference."},
    {"TOPIC": "contribution 3", "CLAIM": "Tests temporal transportability from 2019 to a fully held-out 2023 sample.", "CLASSIFICATION": "SUPPORTED", "NOTES": "Neutral wording, no result leakage."},
    {"TOPIC": "research question 1", "CLAIM": "How do weighted aggregate credit-market state distributions differ between the 2019 baseline and the 2023 wave?", "CLASSIFICATION": "SUPPORTED", "NOTES": "Exact prespecified wording."},
    {"TOPIC": "research question 2", "CLAIM": "How do individual firms transition among credit-market states between 2019 and 2023?", "CLASSIFICATION": "SUPPORTED", "NOTES": "Exact prespecified wording."},
    {"TOPIC": "research question 3", "CLAIM": "Are cumulative COVID-era arrears and liquidity stress associated with constrained non-application in 2023, conditional on baseline credit-market state?", "CLASSIFICATION": "SUPPORTED", "NOTES": "Exact prespecified wording."},
    {"TOPIC": "research question 4", "CLAIM": "How well do credit-state classifiers developed on the 2019 sample transport to the held-out 2023 sample?", "CLASSIFICATION": "SUPPORTED", "NOTES": "Exact prespecified wording."}
]

pd.DataFrame(trace).to_csv("INTRO_LITERATURE_CITATION_TRACEABILITY_LOCKED.csv", index=False)
pd.DataFrame(audit).to_csv("INTRO_LITERATURE_CLAIM_AUDIT_LOCKED.csv", index=False)
