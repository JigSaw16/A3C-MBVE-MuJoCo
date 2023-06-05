# A3C_MBVE_MuJoCo

1. Inicjalizacja:
    - Inicjalizacja globalnego modelu sieci neuronowej.
    - Inicjalizacja globalnego licznika kroków.

2. Asynchroniczne środowiska:

    - Utworzenie kilku asynchronicznych agentów, każdy w swoim środowisku.
    - Przypisanie każdemu agentowi lokalnego modelu sieci neuronowej, który jest klonem globalnego modelu.

3. Asynchroniczne uczenie:

    - Dla każdego agenta:
        - Pobierz stan początkowy ze środowiska.
        - Dopóki nie osiągnięto pewnego warunku końcowego:
            - Wykonaj kroki w środowisku, korzystając z lokalnego modelu sieci neuronowej.
            - Zbierz doświadczenie w postaci obserwacji, nagród i stanów następnych.
            - Aktualizuj lokalny model sieci neuronowej poprzez obliczenie gradientów za pomocą algorytmu propagacji wstecznej i stosowanie ich do lokalnego modelu.
            - Wysyłaj lokalne zmiany modelu do globalnego modelu.
            - Aktualizuj licznik kroków.


4. Aktualizacja globalnego modelu:

    - Jeśli licznik kroków osiągnie określoną wartość, zatrzymaj trenowanie agentów i przeprowadź aktualizację globalnego modelu.
    - Zsumuj lokalne zmiany modelu ze wszystkich agentów.
    - Zastosuj zsumowane zmiany do globalnego modelu za pomocą algorytmu propagacji wstecznej.

5. Powtórz kroki 3-4:
    - Wznów asynchroniczne uczenie agentów, poczynając od aktualizowanego globalnego modelu.