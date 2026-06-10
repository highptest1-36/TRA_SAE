# BTC GRADING — CHI TIẾT ĐẦY ĐỦ (câu hỏi + đáp án + giải thích)

36 câu, IP 115.79.37.109. Sắp theo query_id.

## 1. T1_0008  (type1, 32.2s, HTTP 200)
**Hỏi:** Does having thermal sensors and an optical camera guarantee emergency response data in this case?
**Premises:**
  0. If a satellite has calibrated thermal sensors, then it can monitor surface temperature.
  1. If a satellite can monitor surface temperature and has cloud-penetrating radar, then it can support disaster mapping.
  2. If a satellite supports disaster mapping, then it can provide emergency response data.
  3. Satellite Vega has calibrated thermal sensors.
  4. Satellite Vega does not have cloud-penetrating radar.
  5. Satellite Vega has a high-resolution optical camera.
  6. All satellites with high-resolution optical cameras can capture daytime images.
**Options:** ['Yes', 'No', 'Uncertain']
**→ Answer:** `No`   ·  premises_used=[0, 1, 2, 3, 4]
**Explanation:** Vega lacks cloud-penetrating radar, which is required to support disaster mapping (Premises 1, 2). Without disaster mapping support, it cannot provide emergency response data (Premises 2, 3), regardless of its optical camera (Premises 6, 7).

## 2. T1_0016  (type1, 26.6s, HTTP 200)
**Hỏi:** Does Robot Kappa satisfy every requirement for preparing outbound orders?
**Premises:**
  0. If a warehouse robot has obstacle detection, then it can navigate aisles safely.
  1. If a robot can navigate aisles safely and has barcode scanning, then it can pick inventory items.
  2. If a robot can pick inventory items and has weight verification, then it can prepare outbound orders.
  3. Robot Kappa has obstacle detection.
  4. Robot Kappa has barcode scanning.
  5. Robot Kappa lacks weight verification.
  6. Robot Kappa has a long-life battery.
**Options:** ['Yes', 'No', 'Uncertain']
**→ Answer:** `No`   ·  premises_used=[0, 1, 2, 3, 4, 5]
**Explanation:** Robot Kappa lacks weight verification, which is required to transition from picking inventory to preparing outbound orders, despite having obstacle detection and barcode scanning.

## 3. T1_0022  (type1, 34.1s, HTTP 200)
**Hỏi:** Does the Amber Amulet qualify for public display, according to the premises?
**Premises:**
  0. If an artifact has a humidity-control log and no pest-damage report, then it is storage-ready.
  1. If an artifact is storage-ready and has a provenance certificate, then it is eligible for exhibition.
  2. If an artifact is eligible for exhibition and has curator approval, then it can be placed on public display.
  3. If an artifact is fragile, then it requires low-light protection.
  4. If an artifact requires low-light protection and can be placed on public display, then it must be displayed in a climate-controlled case.
  5. The Amber Amulet has a humidity-control log.
  6. The Amber Amulet has no pest-damage report.
  7. The Amber Amulet has a provenance certificate.
  8. The Amber Amulet has curator approval.
  9. The Amber Amulet is fragile.
**Options:** ['Yes', 'No', 'Uncertain']
**→ Answer:** `Yes`   ·  premises_used=[0, 1, 2, 3, 5, 6, 7, 8, 9]
**Explanation:** The Amber Amulet has a humidity-control log and no pest-damage report, making it storage-ready (Premises 1, 6, 7). It is storage-ready and has a provenance certificate, making it eligible for exhibition (Premises 2, 8). It is eligible for exhibition and has curator approval, allowing it to be placed on public display (Premises 3, 9). While it is fragile and requires low-light protection, the premises confirm it qualifies for public display placement.

## 4. T1_0024  (type1, 34.1s, HTTP 200)
**Hỏi:** Do the premises prove that the Atlas case can be formally closed?
**Premises:**
  0. If a server is patched and its logs have been reviewed, then the incident is contained.
  1. If an incident is contained and all affected passwords have been reset, then the affected account is secured.
  2. If an affected account is secured and the forensic report has been submitted, then the case is audit-ready.
  3. If a case is audit-ready and a manager signs off, then the case can be formally closed.
  4. The Atlas server is patched.
  5. The Atlas server logs have been reviewed.
  6. All affected Atlas passwords have been reset.
  7. The Atlas forensic report has been submitted.
**Options:** ['Yes', 'No', 'Uncertain']
**→ Answer:** `Uncertain`   ·  premises_used=[0, 1, 2, 3, 4, 5, 6, 7]
**Explanation:** Answer: Uncertain.

## 5. T1_0026  (type1, 34.0s, HTTP 200)
**Hỏi:** Is MedKit-7 eligible to use the aerial corridor, according to the premises?
**Premises:**
  0. If a package is medical and weighs under 2 kilograms, then it receives priority delivery status.
  1. If a package has priority delivery status and its route is clear, then it can be dispatched.
  2. If a package can be dispatched and the weather is safe, then it is eligible to use the aerial corridor.
  3. If a package is eligible to use the aerial corridor and an operator is assigned, then launch is approved.
  4. If an emergency waiver is approved and an alternate route is mapped, then the route is clear.
  5. The MedKit-7 package is medical.
  6. The MedKit-7 package weighs under 2 kilograms.
  7. An emergency waiver is approved for MedKit-7.
  8. An alternate route is mapped for MedKit-7.
  9. The weather is safe for MedKit-7.
**Options:** ['Yes', 'No', 'Uncertain']
**→ Answer:** `No`   ·  premises_used=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
**Explanation:** Answer: No.

## 6. T1_0028  (type1, 27.5s, HTTP 200)
**Hỏi:** Do the premises establish that Mira should receive the standard antiviral dose?
**Premises:**
  0. If a patient has a fever and low oxygen saturation, then the patient is placed in respiratory isolation.
  1. If a patient is in respiratory isolation and has a positive rapid test, then the patient enters the antiviral protocol.
  2. If a patient enters the antiviral protocol and has a completed renal screen, then the patient receives the standard antiviral dose.
  3. If a patient has renal impairment, then the patient requires dose adjustment.
  4. If a patient requires dose adjustment, then the patient should not receive the standard antiviral dose without review.
  5. Mira has a fever.
  6. Mira has low oxygen saturation.
  7. Mira has a positive rapid test.
  8. Mira has renal impairment.
**Options:** ['Yes', 'No', 'Uncertain']
**→ Answer:** `No`   ·  premises_used=[0, 1, 3, 4, 5, 6, 7, 8]
**Explanation:** Mira meets the criteria for the antiviral protocol (fever, low oxygen, positive test), but her renal impairment requires dose adjustment, which prevents her from receiving the standard dose without a review. Since no review is indicated, she does not receive the standard dose.

## 7. T1_0030  (type1, 26.7s, HTTP 200)
**Hỏi:** Does the Harbor microgrid have a high resilience score, according to the premises?
**Premises:**
  0. If a solar array is online and the battery bank is charged, then the microgrid can island from the main grid.
  1. If the microgrid can island and the load forecast is stable, then night service can be maintained.
  2. If night service can be maintained and the diesel backup has been tested, then the microgrid has a high resilience score.
  3. If the microgrid has a high resilience score and community notification has been sent, then the emergency hub is ready.
  4. The Harbor microgrid solar array is online.
  5. The Harbor microgrid battery bank is charged.
  6. The Harbor load forecast is stable.
  7. The Harbor diesel backup has been tested.
**Options:** ['Yes', 'No', 'Uncertain']
**→ Answer:** `Yes`   ·  premises_used=[0, 1, 2, 4, 5, 6, 7]
**Explanation:** The premises form a logical chain: (Solar online + Battery charged) → Can island (Premises 1, 5, 6); (Can island + Stable forecast) → Night service maintained (Premises 2, 7); (Night service + Diesel tested) → High resilience score (Premises 3, 8). All conditions for the Harbor microgrid are met.

## 8. T1_0032  (type1, 34.1s, HTTP 200)
**Hỏi:** Do the premises establish that the River Codex is safe for public release?
**Premises:**
  0. If a manuscript is scanned at 600 dpi and its metadata is complete, then it is preservation-ready.
  1. If a manuscript is preservation-ready and its rights are cleared, then it is eligible for the public portal.
  2. If a manuscript is eligible for the public portal and OCR has been verified, then it is searchable online.
  3. If a manuscript contains personal data, then privacy review is required.
  4. If privacy review is required and redaction is complete, then the manuscript is safe for public release.
  5. The River Codex was scanned at 600 dpi.
  6. The River Codex metadata is complete.
  7. The River Codex rights are cleared.
  8. The River Codex OCR has been verified.
  9. The River Codex contains personal data.
**Options:** ['Yes', 'No', 'Uncertain']
**→ Answer:** `Yes`   ·  premises_used=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
**Explanation:** Answer: Yes.

## 9. T1_0034  (type1, 28.3s, HTTP 200)
**Hỏi:** Is a temporary no-take zone recommended for Azure Reef, according to the premises?
**Premises:**
  0. If a reef shows bleaching signs and has a temperature anomaly, then a stress alert is issued.
  1. If a reef has a stress alert and coral cover is below 30 percent, then it becomes a restoration priority.
  2. If a reef is a restoration priority and local fishing pressure is high, then a temporary no-take zone is recommended.
  3. If a temporary no-take zone is recommended and community consultation is complete, then the conservation plan is approved.
  4. Azure Reef shows bleaching signs.
  5. Azure Reef has a temperature anomaly.
  6. Azure Reef has coral cover below 30 percent.
  7. Azure Reef has high local fishing pressure.
**Options:** ['Yes', 'No', 'Uncertain']
**→ Answer:** `Yes`   ·  premises_used=[0, 1, 2, 4, 5, 6, 7]
**Explanation:** The premises logically lead to the conclusion that a temporary no-take zone is recommended for Azure Reef because it shows bleaching signs, has a temperature anomaly, has coral cover below 30 percent, and experiences high local fishing pressure.

## 10. T1_0042  (type1, 23.8s, HTTP 200)
**Hỏi:** Is Asha listed as an active contributor?
**Premises:**
  0. If a researcher completed ethics training and has lab access, then that researcher can handle participant data.
  1. If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.
  2. Every researcher who may join Study Alpha is listed as an active contributor.
  3. Asha completed ethics training.
  4. Asha has lab access.
  5. Asha has supervisor approval.
  6. Study Alpha has 12 enrolled participants.
  7. No premise states whether Asha has budget approval.
**Options:** ['Yes', 'No', 'Uncertain']
**→ Answer:** `Yes`   ·  premises_used=[0, 1, 2, 3, 4, 5]
**Explanation:** Asha completed ethics training and has lab access, so she can handle participant data. With supervisor approval, she may join Study Alpha. Since all researchers who may join Study Alpha are listed as active contributors, Asha is listed as an active contributor.

## 11. T1_0043  (type1, 34.1s, HTTP 200)
**Hỏi:** Does Asha have budget approval?
**Premises:**
  0. If a researcher completed ethics training and has lab access, then that researcher can handle participant data.
  1. If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.
  2. Every researcher who may join Study Alpha is listed as an active contributor.
  3. Asha completed ethics training.
  4. Asha has lab access.
  5. Asha has supervisor approval.
  6. Study Alpha has 12 enrolled participants.
  7. No premise states whether Asha has budget approval.
**Options:** ['Yes', 'No', 'Uncertain']
**→ Answer:** `Uncertain`   ·  premises_used=[0, 1, 2, 3, 5]
**Explanation:** The premises establish that Asha can handle participant data and may join Study Alpha due to completing ethics training, having lab access, and receiving supervisor approval. However, no premise links these conditions to budget approval, nor does any premise mention budget approval at all. Therefore, it is unknown whether Asha has budget approval.

## 12. T1_0048  (type1, 26.9s, HTTP 200)
**Hỏi:** Does Linh have pharmacy training?
**Premises:**
  0. If a clinic volunteer has first-aid certification and completed patient privacy training, then the volunteer may assist at the triage desk.
  1. If a volunteer may assist at the triage desk and has morning availability, then the volunteer is assigned to the morning triage shift.
  2. Every volunteer assigned to the morning triage shift receives a blue access badge.
  3. Linh has first-aid certification.
  4. Linh completed patient privacy training.
  5. Linh has morning availability.
  6. The morning triage shift needs 3 volunteers.
  7. No premise states that Linh has pharmacy training.
**Options:** ['Yes', 'No', 'Uncertain']
**→ Answer:** `No`   ·  premises_used=[3, 4, 5]
**Explanation:** The premises do not state that Linh has pharmacy training, nor do they imply it through any logical chain. The relevant premises (1, 2, 4, 5, 6) concern first-aid certification, patient privacy training, and availability for the triage shift, but none mention pharmacy training.

## 13. T2_0001  (type2, 33.9s, HTTP 200)
**Hỏi:** A capacitor has capacitance C = 47 μF and is connected to a potential difference U = 12 V. Calculate the energy stored in the capacitor.
**→ Answer:** `6`
**Explanation:** Answer: 6.

## 14. T2_0002  (type2, 34.1s, HTTP 200)
**Hỏi:** A capacitor stores a charge Q = 2.4 mC when connected to a voltage U = 16 V. Calculate the capacitance of the capacitor.
**→ Answer:** `1.5e-4` F
**Explanation:** Answer: 1.5e-4 F.

## 15. T2_0003  (type2, 34.0s, HTTP 200)
**Hỏi:** Two point charges q1 = +3.0 μC and q2 = -5.0 μC are separated by a distance of 0.30 m in air. Calculate the magnitude of the electrostatic force between them. Use k = 9.0 × 10^9 N·m²/C².
**→ Answer:** `0.135`
**Explanation:** Answer: 0.135.

## 16. T2_0004  (type2, 34.1s, HTTP 200)
**Hỏi:** Two charges +4.0 nC and -4.0 nC are fixed 20 cm apart in air. Calculate the magnitude of the electric field at the midpoint between the charges. Use k = 9.0 × 10^9 N·m²/C².
**→ Answer:** `9000000000`
**Explanation:** Answer: 9000000000.

## 17. T2_0005  (type2, 34.1s, HTTP 200)
**Hỏi:** A point P is 10 cm from a charge q1 = +6.0 nC and 20 cm from a charge q2 = -2.0 nC. Calculate the electric potential at P. Use k = 9.0 × 10^9 N·m²/C².
**→ Answer:** `9`
**Explanation:** Answer: 9.

## 18. T2_0006  (type2, 34.0s, HTTP 200)
**Hỏi:** Two capacitors C1 = 6.0 μF and C2 = 3.0 μF are connected in series across an 18 V battery. Calculate the charge stored on each capacitor.
**→ Answer:** `18` V
**Explanation:** Answer: 18 V.

## 19. T2_0007  (type2, 34.0s, HTTP 200)
**Hỏi:** A 24 Ω resistor is connected to a 12 V battery for 5 minutes. Calculate the electrical energy converted in the resistor.
**→ Answer:** `1800`
**Explanation:** Answer: 1800.

## 20. T2_0008  (type2, 27.6s, HTTP 200)
**Hỏi:** Two resistors R1 = 12 Ω and R2 = 18 Ω are connected in series to a 15 V source. Calculate the current in the circuit.
**→ Answer:** `0.5` A
**Explanation:** In a series circuit, the total resistance is the sum of the individual resistances (12 Ω + 18 Ω = 30 Ω). Using Ohm's Law (I = U/R), the current is calculated as 15 V / 30 Ω = 0.5 A.

## 21. T2_0009  (type2, 29.1s, HTTP 200)
**Hỏi:** Two resistors R1 = 6 Ω and R2 = 3 Ω are connected in parallel across a 12 V source. Calculate the total current drawn from the source.
**→ Answer:** `6` A
**Explanation:** The current through R1 is 12/6 = 2 A, and the current through R2 is 12/3 = 4 A. Since they are in parallel, the total current is the sum of the branch currents: 2 A + 4 A = 6 A.

## 22. T2_0011  (type2, 34.1s, HTTP 200)
**Hỏi:** An inductor with inductance L = 0.40 H is connected to an AC source of frequency f = 60 Hz. Calculate the inductive reactance.
**→ Answer:** `48`
**Explanation:** Answer: 48.

## 23. T2_0012  (type2, 29.3s, HTTP 200)
**Hỏi:** A car starts with an initial speed of 8 m/s and travels 66 m in 6 s with constant acceleration. Calculate the acceleration of the car.
**→ Answer:** `1` m/s
**Explanation:** Using the formula $s = ut + \frac{1}{2}at^2$ with $s=66$, $u=8$, and $t=6$, we solve for $a$ and find $a=1$.

## 24. T2_0013  (type2, 23.2s, HTTP 200)
**Hỏi:** A car moving at 20 m/s brakes uniformly to rest with acceleration -4.0 m/s². Calculate the braking distance.
**→ Answer:** `50`
**Explanation:** Using the kinematic equation v² = u² + 2as with v=0, u=20, and a=-4.0 yields s = 50 m.

## 25. T2_0014  (type2, 34.0s, HTTP 200)
**Hỏi:** A person of mass 60 kg stands in an elevator accelerating upward at 1.5 m/s². Calculate the normal force exerted by the elevator floor on the person. Use g = 9.8 m/s².
**→ Answer:** `678`
**Explanation:** The normal force is calculated using the formula $N = m(g + a)$ for an upward acceleration. Substituting $m=60$ kg, $g=9.8$ m/s², and $a=1.5$ m/s² yields $N = 60 \times 11.3 = 678$ N.

## 26. T2_0015  (type2, 34.1s, HTTP 200)
**Hỏi:** A 2.0 kg object starts from rest. A constant net force of 10 N acts on it over a distance of 5.0 m. Calculate the final speed of the object.
**→ Answer:** `50`
**Explanation:** Answer: 50.

## 27. T2_0016  (type2, 34.0s, HTTP 200)
**Hỏi:** A ball is thrown vertically upward with an initial speed of 19.6 m/s. Calculate the maximum height reached by the ball. Use g = 9.8 m/s².
**→ Answer:** `2` gh
**Explanation:** Answer: 2 gh.

## 28. T2_0017  (type2, 34.0s, HTTP 200)
**Hỏi:** How much heat is required to raise the temperature of 0.50 kg of water by 15 °C? Use c = 4200 J/(kg·°C).
**→ Answer:** `31500`
**Explanation:** Answer: 31500.

## 29. T2_0018  (type2, 34.0s, HTTP 200)
**Hỏi:** Calculate the heat required to melt 0.20 kg of ice at 0 °C. Use the latent heat of fusion of ice L = 3.34 × 10^5 J/kg.
**→ Answer:** `66800` J
**Explanation:** The question asks for the heat required to melt a specific mass of ice at its melting point.
1. Identify the given values:
   - Mass of ice ($m$) = 0.20 kg
   - Latent heat of fusion of ice ($L$) = $3.34 \times 10^5$ J/kg
   - The ice is already at 0 °C, so no heat is needed to raise the temperature, only to change the phase from solid to liquid.

2. Select the appropriate formula:
   The heat energy ($Q$) required for a phase change (melting) is calculated using:
   $$Q = m \times L$$

3. Perform the calculation:
   $$Q = 0.20 \, \text{kg} \times 3.34 \times 10^5 \, \text{J/kg}$$
   $$Q = 0.2 \times 334000$$
   $$Q = 66800 \, \text{J}$$

4. Convert to scientific notation (optional but standard):
   $$Q = 6.68 \times 10^4 \, \text{J}$$

The calculation is straightforward multiplication. The result represents the energy needed to convert the solid ice into liquid water at the same temperature.

## 30. T2_0019  (type2, 34.0s, HTTP 200)
**Hỏi:** A gas sample contains n = 0.50 mol at temperature T = 300 K in a volume V = 0.010 m³. Calculate the pressure of the gas. Use R = 8.314 J/(mol·K).
**→ Answer:** `7`
**Explanation:** Answer: 7.

## 31. T2_0020  (type2, 34.1s, HTTP 200)
**Hỏi:** An object is placed 30 cm in front of a converging lens with focal length 20 cm. Calculate the image distance.
**→ Answer:** `20` cm
**Explanation:** Answer: 20 cm.

## 32. T2_0028  (type2, 15.3s, HTTP 200)
**Hỏi:** A lamp carries current I = 0.25 A under voltage U = 12 V. Calculate its power.
**→ Answer:** `3` W
**Explanation:** Electric power is calculated by multiplying the voltage across the component by the current flowing through it (P = U × I). Here, 12 V multiplied by 0.25 A equals 3 Watts.

## 33. T2_0029  (type2, 34.1s, HTTP 200)
**Hỏi:** A charge q = 3 uC moves through a potential difference U = 5 V. Calculate the work done.
**→ Answer:** `1.5e-5` J
**Explanation:** 1.  **Identify the given values:**
    *   Charge ($q$) = $3 \, \mu\text{C} = 3 \times 10^{-6} \, \text{C}$.
    *   Potential difference ($U$) = $5 \, \text{V}$.

2.  **Recall the formula for electric work:**
    The work done ($A$ or $W$) by the electric field when a charge moves through a potential difference is given by:
    $$A = q \cdot U$$

3.  **Perform the calculation:**
    $$A = (3 \times 10^{-6} \, \text{C}) \times (5 \, \text{V})$$
    $$A = 15 \times 10^{-6} \, \text{J}$$
    $$A = 1.5 \times 10^{-5} \, \text{J}$$

4.  **Final check:**
    The unit of work is Joules (J), which is consistent with Coulombs $\times$ Volts. The magnitude is correct for micro-Coulombs and Volts.

## 34. T2_0039  (type2, 34.0s, HTTP 200)
**Hỏi:** A conductor has length l = 2 m, cross-sectional area S = 1 mm^2, and resistivity rho = 0.5 ohm*mm^2/m. Calculate its resistance.
**→ Answer:** `1` ohm
**Explanation:** 1.  **Identify the given values:**
    *   Length ($l$) = 2 m
    *   Cross-sectional area ($S$) = 1 mm²
    *   Resistivity ($\rho$) = 0.5 Ω·mm²/m

2.  **Recall the formula for electrical resistance:**
    The resistance $R$ of a conductor is calculated using the formula:
    $$R = \rho \cdot \frac{l}{S}$$

3.  **Check unit consistency:**
    *   $\rho$ is given in Ω·mm²/m.
    *   $l$ is given in m.
    *   $S$ is given in mm².
    The units match perfectly with the formula (no conversion needed).

4.  **Substitute the values into the formula:**
    $$R = 0.5 \cdot \frac{2}{1}$$

5.  **Perform the calculation:**
    $$R = 0.5 \cdot 2 = 1$$

6.  **Determine the unit of the result:**
    Since $\rho$ is in Ohms (Ω), the result $R$ is in Ohms (Ω).

7.  **Final Answer:** 1 Ω.

## 35. T2_0049  (type2, 34.0s, HTTP 200)
**Hỏi:** A transformer has primary turns N1 = 200, secondary turns N2 = 800, and primary voltage U1 = 12 V. Calculate the secondary voltage.
**→ Answer:** `48` V
**Explanation:** 1.  **Identify the principle**: A transformer operates on the principle of electromagnetic induction, where the ratio of the voltages across the primary and secondary coils is directly proportional to the ratio of their number of turns.
2.  **State the formula**: The relationship is given by:
    $$ \frac{U_1}{N_1} = \frac{U_2}{N_2} $$
    Where:
    *   $U_1$ is the primary voltage.
    *   $N_1$ is the number of primary turns.
    *   $U_2$ is the secondary voltage (unknown).
    *   $N_2$ is the number of secondary turns.
3.  **Rearrange the formula**: Solve for $U_2$:
    $$ U_2 = U_1 \times \frac{N_2}{N_1} $$
4.  **Substitute the values**:
    *   $U_1 = 12 \text{ V}$
    *   $N_1 = 200$
    *   $N_2 = 800$
    $$ U_2 = 12 \times \frac{800}{200} $$
5.  **Calculate**:
    *   $\frac{800}{200} = 4$
    *   $U_2 = 12 \times 4 = 48$
6.  **Conclusion**: The secondary voltage is 48 V.

## 36. T2_0050  (type2, 33.8s, HTTP 200)
**Hỏi:** A resistor dissipates P = 9 W when the current is I = 3 A. Calculate its resistance.
**→ Answer:** `1` ohm
**Explanation:** Using the power formula $P = I^2R$, we substitute $P=9$ and $I=3$ to get $9 = 3^2 \cdot R$, which simplifies to $9 = 9R$. Solving for $R$ yields $1 \Omega$.
