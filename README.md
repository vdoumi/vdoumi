# V-Doumi

I'm Virtual Math Helper, What can I help you?

## FEATURES

-   AI 버튜버 인강 강사이다.
-   수연도우미처럼 유튜브 영상으로 개념 설명 및 단원별 문제를 풀이한다.
-   문제 사진을 올렸을 때 해당 문제를 설명하는 영상을 출력한다.

## TODOS

-   [x] 수학 예시 문제를 curating한 후에 pdf/image/latex 의 이해도 정도를 비교한다.
    -   [x] 여러 number의 수학 문제가 내용이 연결되어 있는 경우를 파악하여 세트로 묶는다. (나중에 한 영상에서 설명할 수 있게)
    -   [ ] (나중에) 문제 전체를 한 번에 찍으면 자동으로 crop하여 분리, 그 후 sorter 실행
-   [x] 문제를 푸는 파이프라인을 설계하고 solver/scripter 최소 MVP를 구현한다.
-   [x] 깔끔하게 푼 풀이를 바탕으로 Script를 작성한다.
-   [x] Script를 바탕으로 PPT 등의 시각 자료를 생성한다.
    -   [x] Marp 사용해서 시각 자료 제작
        -   [x] 시각 자료 깨짐 문제 해결
    -   [ ] Manim과 같은 animation library에 대해서도 적극적인 차용이 필요함.
    -   [ ] 손글씨나 펜으로 그림을 그리는 기능까지 있으면... 좋지.
-   [x] 적절한 TTS API나 오픈소스를 찾아서 음성을 만든다.
-   [x] audio timeline을 가지고 버튜버를 rendering한다.
-   [ ] 영상을 생성한다.
