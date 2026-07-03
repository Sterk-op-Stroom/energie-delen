# Dashboard — Gebruikersgids

Het Energie Delen dashboard is een browserinterface voor het uitvoeren van energiedeelsimulaties. Geen programmeerkennis nodig.

---

## Installatie en opstarten

### Stap 1 — Download het project

[Download de zip](https://github.com/Sterk-op-Stroom/energie-delen/archive/refs/heads/main.zip) en pak hem uit op een plek naar keuze.

### Stap 2 — Start het dashboard

Ga naar de uitgepakte map en open de `simulation/` map. Start het dashboard:

**Windows:** dubbelklik op `launch_dashboard.bat`  
**macOS / Linux:** voer `./launch_dashboard.sh` uit in de terminal

Het dashboard opent automatisch in je browser op `http://localhost:5006`.

---

## Overzicht

Het dashboard heeft drie pagina's die je in volgorde doorloopt: **Data-invoer → Simulatie → Resultaten**.

---

## Pagina 1 — Data-invoer

### Data laden

Je hebt twee soorten data nodig:

- **Prosumerdata** — slimme metergegevens per huishouden (kWh; positief = verbruik, negatief = teruglevering)
- **Productie-assets** — opbrengst van gedeelde opwekinstallaties (zonnepanelen, windturbine)

Beide moeten in Parquet-formaat zijn. Zie `docs/data_formats.md` voor het vereiste schema.

Je hebt drie manieren om data te laden:

| Optie | Wanneer |
|---|---|
| **Laad voorbeelddata** | Geen eigen data beschikbaar; genereert 5 prosumers en 2 assets over 7 dagen om het dashboard te verkennen |
| **Pad invoeren** | Je hebt Parquet-bestanden op schijf staan |
| **Bestand uploaden** | Je wilt een bestand slepen of selecteren vanuit je bestandsbeheer |

Elk geüpload bestand wordt aan een lijst toegevoegd — uploaden overschrijft niet. Vouw **Geselecteerde bestanden** uit om te zien welke bestanden er staan. Per bestand kun je:
- het **vinkje** aan- of uitzetten om te bepalen of het meegenomen wordt
- het **prullenbakpictogram** (🗑) gebruiken om het uit de lijst te verwijderen

Als je meerdere bestanden per rol aanvinkt (bijv. twee prosumerbestanden), worden ze automatisch samengevoegd. Dit is handig als je data per maand of per meter in aparte bestanden hebt.

### Inspecteren

Klik op **Laad & inspecteer**. Het dashboard toont:

- Een tabel met alle meters: ID, tijdsbereik, aantal datapunten, frequentie en percentage ontbrekende waarden
- Een **tijdlijn** die laat zien hoeveel meters op elk moment data hebben
- Een **dekkingsheatmap** per meter (groen = volledig, rood = gaten) — schakelbaar tussen dag/week/maand/kwartaal
- Een **ontbrekend %**-grafiek die meters rangschikt op hoeveel data er ontbreekt

Boven de grafieken zie je het **Inspectierapport**:

| Veld | Betekenis |
|---|---|
| Voorgestelde start / einde | Langste periode waarop alle meters overlap hebben |
| Voorgestelde frequentie | Meest voorkomend interval over alle meters |
| Frequentie consistent | Of alle meters hetzelfde interval gebruiken (`ja` / `nee ⚠`) |

Klik op **Volgende: Simulatie-instellingen →** om verder te gaan. De voorgestelde waarden worden automatisch ingevuld op de volgende pagina.

---

## Pagina 2 — Simulatie

### Instellingen

**Datumbereik** — start- en einddatum van de simulatie (DD-MM-JJJJ). Standaard de langste overlap uit de inspectie.

Het dashboard werkt met drie lagen van prijzen. Elke laag is optioneel bovenop de vorige:

**1 · Lokale prijs (EUR/kWh)** — het tarief dat prosumers onderling betalen voor lokaal gedeelde energie. Dit is de kern van het P2P-model: alleen de kWh die daadwerkelijk lokaal is gedeeld, wordt hiermee geprijsd. Doorgaans lager dan het markttarief; typisch rond de €0,075 voor Nederlandse energiecoöperaties.

**2 · Prijsmodel markt** — wat er met de resterende netstromen gebeurt *na* lokaal delen. Niet al het verbruik kan lokaal worden gedekt (er is altijd netimport), en niet alle opwek wordt lokaal verbruikt (er is altijd netexport). Deze laag voegt de bijbehorende kosten en opbrengsten toe:
- **Geen** — alleen lokale kosten worden berekend
- **Vaste prijs** — vul een importprijs (wat leden betalen voor resterende netimport) en een exportvergoeding (wat de coöperatie ontvangt voor resterende netexport) in

  Samen met de lokale prijs geeft dit het complete kostenplaatje *met* lokaal delen: lokale kosten + netto marktkosten.

**3 · Prijsmodel energieleverancier (vergelijking)** — de counterfactual: wat hadden leden betaald als er *geen* lokaal delen was geweest? Dan zou al het verbruik van het net zijn gekomen en alle opwek direct zijn teruggeleverd. Door dezelfde importprijs en exportvergoeding als bij het markttarief in te vullen, zie je naast de werkelijke kosten ook wat de situatie zonder coöperatie zou zijn geweest. Het verschil tussen de twee is de besparing van het energiedelen.

**Geavanceerde instellingen** (ingeklapt):

- **Frequentie** — tijdstapresolutie (bijv. 15 min). Standaard automatisch gedetecteerd vanuit de data.
- **Ontbrekende data** — hoe gaten in meterdata worden behandeld:

  | Optie | Gedrag |
  |---|---|
  | `fill_zero` | Ontbrekende tijdstappen tellen als nul |
  | `fill_forward` | Gaten worden gevuld met de laatste bekende waarde — zinvol als meters soms uitvallen maar het verbruik doorgaat |
  | `keep_nan` | Gaten blijven NaN en propageren door de pipeline |
  | `error` | De simulatie stopt direct als er ontbrekende data is |

- **NaN-beleid** — hoe NaN-waarden meewegen in de aggregatie:

  | Optie | Gedrag |
  |---|---|
  | `treat_as_zero` | NaN-waarden tellen als nul in de som |
  | `propagate` | Eén NaN op een tijdstap maakt het totaal van dat tijdstap NaN |

### Uitvoeren

Klik op **Simulatie starten**. Het **Pipeline-logboek** toont de voortgang per stap. Bij succes verschijnen de KPI-kaarten:

**Energie (kWh):** totale vraag, totaal aanbod, lokaal toegewezen, netimport, netexport  
**Efficiëntie (%):** zelfvoorzieningspercentage (hoeveel vraag lokaal werd gedekt), zelfconsumptiepercentage (hoeveel lokale opwek intern werd verbruikt)  
**Kosten (EUR):** bedrag voor lokaal gedeelde energie (lokale prijs × lokaal toegewezen kWh)  
**Marktkosten (EUR):** zichtbaar als marktprijzen zijn ingesteld — importkosten, exportopbrengst en netto marktkosten voor de resterende netstromen na lokaal delen  
**Vergelijking energieleverancier (EUR):** zichtbaar als de counterfactual is ingesteld — dezelfde drie cijfers, maar dan voor de situatie zonder lokaal delen; het verschil met de marktkosten is de besparing

Klik op **Bekijk resultaten →** om door te gaan.

---

## Pagina 3 — Resultaten

### Verkennen

Twee knoppen gelden voor alle grafieken:
- **Datumbereik** — zoom in op een deelperiode van het gesimuleerde venster
- **Aggregatie** — bekijk ruwe data (15 min) of resample naar uur / dag / week

De grafieken zijn verdeeld over vier tabbladen:

**Energiestromen** — aanbod vs. vraag, en de verdeling over lokaal gedeeld / netimport / netexport  
**Zelfvoorzienendheid & consumptie** — zelfvoorzienings- en zelfconsumptiepercentage over tijd, plus toewijzing per prosumer (bij ≤ 20 prosumers)  
**Kosten** — gemeenschapskosten en kosten per prosumer; marktkosten als marktprijzen zijn ingesteld  
**Gemiddeld profiel** — aggregeert alle tijdstappen naar tijdstip van de dag, dag van de week of dag van het jaar, om structurele patronen zichtbaar te maken (bijv. ochtend-/avondpieken, seizoensvariatie)

### Prosumertabel

Een tabel met één rij per prosumer: toegewezen kWh, netimport, zelfvoorzieningsgraad en kosten lokaal delen. Bij marktprijzen komen hier drie kolommen bij: marktimportkosten, marktexportopbrengst en netto marktkosten.

Twee downloadknoppen:
- **Download prosumer-CSV** — geaggregeerde totalen per prosumer over de hele periode
- **Download tijdreeks-CSV** — volledige tijdstapdata per prosumer

---

## Veelgestelde problemen

| Symptoom | Oplossing |
|---|---|
| Resultaten-knop is grijs | Voer eerst een simulatie uit op pagina 2 |
| "Geen overlap"-waarschuwing | Prosumer- en productiebestanden dekken verschillende periodes; controleer de datumbereiken |
| Frequentie inconsistent ⚠ | Niet alle meters hebben hetzelfde interval; gebruik alleen bestanden met dezelfde frequentie |
| Andere instellingen proberen | Ga terug naar pagina 2, pas aan en klik opnieuw op **Simulatie starten** — resultaten worden direct bijgewerkt |
