import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PruningMetricCardsComponent } from './pruning-metric-cards.component';

describe('PruningMetricCardsComponent', () => {
  let component: PruningMetricCardsComponent;
  let fixture: ComponentFixture<PruningMetricCardsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PruningMetricCardsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PruningMetricCardsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
